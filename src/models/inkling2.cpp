#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include <cstring>

static bool inkling_enabled(const char * key) {
    const char * value = getenv(key);
    return !value || atoi(value) != 0;
}

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    // the attention input projections are concatenated into one matrix per layer after loading (see load_arch_tensors_post),
    // when they share the type; LLAMA_INKLING_FUSED_QKV=0 disables, a memory-fit pass loads them as they are
    fuse_weights = [&] {
        const char * env = getenv("LLAMA_INKLING_FUSED_QKV");
        return (env == nullptr || atoi(env) != 0) && !ml->no_alloc;
    }();
    const bool fuse_attn = fuse_weights;
    // The fused decoder stores KV in model coordinates. Batched prefill must
    // use those coordinates too, including when the cache is quantized.
    if (fuse_attn && inkling_enabled("LLAMA_INKLING_FUSED") && inkling_enabled("LLAMA_INKLING_ATTN_OP")) {
        attn_rot = false;
    }
    fused_attn.assign(n_layer_phys(), {});
    attn_inp_norm_qkv.assign(n_layer_phys(), nullptr);
    attn_consts.assign(n_layer_phys(), nullptr); // filled by load_arch_tensors_post (not run by a memory-fit pass)
    ffn_consts.assign(n_layer_phys(), nullptr);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // the row groups of the fused projection matrix, in order
        auto & fa = fused_attn[i];
        std::vector<std::pair<std::string, int64_t>> pieces = {
            { tn(LLM_TENSOR_ATTN_Q,         "weight", i).str(), n_embd_head*n_head },
            { tn(LLM_TENSOR_ATTN_REL,       "weight", i).str(), n_rel*n_head },
            { tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i).str(), n_head },
            { tn(LLM_TENSOR_ATTN_GATE,      "weight", i).str(), n_embd_head*n_head },
            { tn(LLM_TENSOR_ATTN_K,         "weight", i).str(), n_embd_head*n_head_kv },
        };
        if (!hparams.is_swa(i)) {
            pieces.emplace_back(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i).str(), n_conv*n_head_kv);
        }
        pieces.emplace_back(tn(LLM_TENSOR_ATTN_V, "weight", i).str(), n_embd_head*n_head_kv);
        if (!hparams.is_swa(i)) {
            pieces.emplace_back(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i).str(), n_conv*n_head_kv);
        }

        bool fuse = fuse_attn;
        ggml_type type = GGML_TYPE_COUNT;
        for (const auto & [name, n_rows] : pieces) {
            const ggml_tensor * meta = ml->get_tensor_meta(name.c_str());
            if (meta == nullptr || meta->ne[0] != n_embd || meta->ne[1] != n_rows) {
                fuse = false;
                break;
            }
            if (type == GGML_TYPE_COUNT) {
                type = meta->type;
            } else if (type != meta->type) {
                fuse = false;
                break;
            }
        }
        if (fuse) {
            fa.pieces = pieces;
        }
        const int skip = fuse ? TENSOR_SKIP | TENSOR_SKIP_QUIET : 0;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, skip);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, skip);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, skip);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, skip);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, skip);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, skip);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, skip);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, skip);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

ggml_context * llama_model_inkling2::fused_context() {
    if (!fused_ctx) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(64*n_layer_phys() + 64),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        fused_ctx.reset(ggml_init(params));
        if (!fused_ctx) {
            throw std::runtime_error("inkling2: failed to create the context of the fused weights");
        }
    }
    return fused_ctx.get();
}

// allocate the assembled tensors next to the layer weights and read their pieces from the file
void llama_model_inkling2::load_fused(llama_model_loader & ml) {
    if (fused_pending.empty()) {
        return;
    }

    std::map<ggml_backend_buffer_type_t, std::vector<ggml_tensor *>> by_buft;
    for (const auto & f : fused_pending) {
        GGML_ASSERT(f.ref != nullptr && f.ref->buffer != nullptr);
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(f.ref->buffer);
        if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                buft = ggml_backend_dev_buffer_type(dev);
            }
        }
        by_buft[buft].push_back(f.t);
    }

    for (auto & [buft, tensors] : by_buft) {
        ggml_init_params params_buf = {
            /*.mem_size   =*/ ggml_tensor_overhead()*tensors.size(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_buf { ggml_init(params_buf) };
        std::vector<ggml_tensor *> dups;
        for (ggml_tensor * t : tensors) {
            ggml_tensor * d = ggml_dup_tensor(ctx_buf.get(), t);
            ggml_set_name(d, ggml_get_name(t));
            dups.push_back(d);
        }
        ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_buf.get(), buft) };
        if (!buf) {
            throw std::runtime_error(format("inkling2: unable to allocate the %s buffer of the fused weights", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t j = 0; j < tensors.size(); ++j) {
            tensors[j]->buffer = dups[j]->buffer;
            tensors[j]->data   = dups[j]->data;
        }
        fused_bufs.emplace_back(std::move(buf));
    }

    std::vector<uint8_t> staging;
    for (const auto & f : fused_pending) {
        if (f.zero_fill) {
            staging.assign(ggml_nbytes(f.t), 0);
            ggml_backend_tensor_set(f.t, staging.data(), 0, staging.size());
        }
        for (const auto & [name, offs] : f.pieces) {
            const llama_model_loader::llama_tensor_weight & w = ml.require_weight(name.c_str());
            const size_t nbytes = ggml_nbytes(w.tensor);

            // read from the file: the loader unmaps the regions of the skipped tensors
            staging.resize(nbytes);
            const auto & file = ml.files.at(w.idx);
            file->seek(w.offs, SEEK_SET);
            file->read_raw(staging.data(), nbytes);
            if (w.tensor->type == f.t->type) {
                GGML_ASSERT(offs + nbytes <= ggml_nbytes(f.t));
                ggml_backend_tensor_set(f.t, staging.data(), offs, nbytes);
            } else {
                // an F32 piece of an F16 assembled tensor is converted here
                GGML_ASSERT(w.tensor->type == GGML_TYPE_F32 && f.t->type == GGML_TYPE_F16);
                const int64_t n = ggml_nelements(w.tensor);
                GGML_ASSERT(offs + n*sizeof(ggml_fp16_t) <= ggml_nbytes(f.t));
                std::vector<ggml_fp16_t> h(n);
                ggml_fp32_to_fp16_row((const float *) staging.data(), h.data(), n);
                ggml_backend_tensor_set(f.t, h.data(), offs, n*sizeof(ggml_fp16_t));
            }
        }
    }

    LLAMA_LOG_INFO("%s: assembled %zu fused weight tensors\n", __func__, fused_pending.size());
    fused_pending.clear();
}

// concatenate the attention input projections of each layer into one matrix, loaded straight from the file,
// and point the layer's tensors at the row groups of that matrix
void llama_model_inkling2::load_arch_tensors_post(llama_model_loader & ml) {
    const int64_t n_embd = hparams.n_embd;

    const char * peer_name = getenv("IK_PEER_HEAD");
    if (peer_name && output && output->ne[1] % 2 == 0) {
        auto * peer = ggml_backend_dev_by_name(peer_name);
        if (!peer || ggml_backend_dev_type(peer) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                peer == ggml_backend_buft_get_device(ggml_backend_buffer_get_type(output->buffer))) {
            throw std::runtime_error("IK_PEER_HEAD requires a second GPU");
        }
        const int64_t rows = output->ne[1]/2;
        auto * ctx = fused_context();
        output_half[0] = ggml_view_2d(ctx, output, n_embd, rows, output->nb[1], 0);
        auto * upper = ggml_view_2d(ctx, output, n_embd, rows, output->nb[1], rows*output->nb[1]);
        output_half[1] = ggml_dup_tensor(ctx, upper);
        ggml_set_name(output_half[1], "output.peer");
        auto * buft = ggml_backend_dev_buffer_type(peer);
        ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, output_half[1])) };
        if (!buf || ggml_backend_view_init(output_half[0]) != GGML_STATUS_SUCCESS || ggml_backend_view_init(upper) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(buf.get(), output_half[1], ggml_backend_buffer_get_base(buf.get())) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("inkling2: failed to allocate peer output rows");
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_copy(upper, output_half[1]);
        fused_bufs.emplace_back(std::move(buf));
        LLAMA_LOG_INFO("%s: %lld output rows on %s\n", __func__, (long long) rows, peer_name);
    }

    // the fused projection matrices
    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & fa = fused_attn[i];
        if (fa.pieces.empty()) {
            continue;
        }
        const ggml_type type = ml.get_tensor_meta(fa.pieces[0].first.c_str())->type;
        int64_t n_rows = 0;
        for (const auto & piece : fa.pieces) {
            n_rows += piece.second;
        }
        fa.w = ggml_new_tensor_2d(fused_context(), type, n_embd, n_rows);
        ggml_format_name(fa.w, "blk.%d.attn_fused.weight", i);

        fused_tensor f;
        f.t   = fa.w;
        f.ref = layers[i].wo;
        const size_t row_size = ggml_row_size(type, n_embd);
        int64_t row = 0;
        for (const auto & [name, n_piece] : fa.pieces) {
            f.pieces.emplace_back(name, row*row_size);
            row += n_piece;
        }
        fused_pending.push_back(std::move(f));
    }

    load_fused(ml);

    // the layer tensors of the pieces become views of the fused matrices
    int n_fused = 0;
    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & fa = fused_attn[i];
        if (fa.pieces.empty()) {
            continue;
        }
        n_fused++;
        auto & layer = layers[i];

        const size_t row_size = ggml_row_size(fa.w->type, n_embd);

        int64_t row = 0;
        for (const auto & [name, n_piece] : fa.pieces) {
            ggml_tensor * view = ggml_view_2d(fused_context(), fa.w, n_embd, n_piece, row_size, row*row_size);
            ggml_set_name(view, name.c_str());
            tensors_by_name.emplace_back(name, view);

            if (name == tn(LLM_TENSOR_ATTN_Q, "weight", i).str()) {
                layer.wq = view;
            } else if (name == tn(LLM_TENSOR_ATTN_REL, "weight", i).str()) {
                layer.attn_rel = view;
                fa.off_rel = row;
            } else if (name == tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i).str()) {
                layer.attn_sink_proj = view;
                fa.off_sink = row;
            } else if (name == tn(LLM_TENSOR_ATTN_GATE, "weight", i).str()) {
                layer.wqkv_gate = view;
                fa.off_gate = row;
                fa.n_q = row + n_piece;
            } else if (name == tn(LLM_TENSOR_ATTN_K, "weight", i).str()) {
                layer.wk = view;
                fa.n_k = n_piece;
            } else if (name == tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i).str()) {
                layer.attn_k_conv_proj = view;
                fa.off_kproj = row - fa.n_q;
                fa.n_k += n_piece;
            } else if (name == tn(LLM_TENSOR_ATTN_V, "weight", i).str()) {
                layer.wv = view;
                fa.n_v = n_piece;
            } else if (name == tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i).str()) {
                layer.attn_v_conv_proj = view;
                fa.off_vproj = row - fa.n_q - fa.n_k;
                fa.n_v += n_piece;
            } else {
                GGML_ABORT("inkling2: unexpected fused piece %s", name.c_str());
            }

            row += n_piece;
        }
        GGML_ASSERT(row == fa.w->ne[1] && fa.n_q + fa.n_k + fa.n_v == row);

        fa.q = ggml_view_2d(fused_context(), fa.w, n_embd, fa.n_q, row_size, 0);
        fa.k = ggml_view_2d(fused_context(), fa.w, n_embd, fa.n_k, row_size, fa.n_q*row_size);
        fa.v = ggml_view_2d(fused_context(), fa.w, n_embd, fa.n_v, row_size, (fa.n_q + fa.n_k)*row_size);
        ggml_format_name(fa.q, "blk.%d.attn_fused_q.weight", i);
        ggml_format_name(fa.k, "blk.%d.attn_fused_k.weight", i);
        ggml_format_name(fa.v, "blk.%d.attn_fused_v.weight", i);

        // the q, k, v input norms of a full-attention layer, stacked for the batched controllers
        if (!hparams.is_swa(i) && layer.attn_q_inp_norm && layer.attn_k_inp_norm && layer.attn_v_inp_norm) {
            ggml_tensor * stacked = ggml_new_tensor_2d(fused_context(), GGML_TYPE_F32, n_embd, 3);
            ggml_format_name(stacked, "blk.%d.attn_inp_norm_qkv.weight", i);
            fused_tensor f;
            f.t   = stacked;
            f.ref = layers[i].wo;
            f.pieces.emplace_back(tn(LLM_TENSOR_ATTN_Q_INP_NORM, "weight", i).str(), 0*n_embd*sizeof(float));
            f.pieces.emplace_back(tn(LLM_TENSOR_ATTN_K_INP_NORM, "weight", i).str(), 1*n_embd*sizeof(float));
            f.pieces.emplace_back(tn(LLM_TENSOR_ATTN_V_INP_NORM, "weight", i).str(), 2*n_embd*sizeof(float));
            fused_pending.push_back(std::move(f));
            attn_inp_norm_qkv[i] = stacked;
        }
    }

    // the constants of the fused decode attention op
    if (fuse_weights) {
        for (int i = 0; i < n_layer_phys(); ++i) {
            const auto & layer = layers[i];
            const ggml_tensor * pieces[5] = { layer.attn_q_clip, layer.attn_rel_clip, layer.attn_k_conv, layer.attn_v_conv, layer.attn_rel_bias };
            bool ok = fused_attn[i].w != nullptr;
            size_t n = 0;
            for (const ggml_tensor * t : pieces) {
                ok = ok && t != nullptr && t->type == GGML_TYPE_F32;
                n += t ? ggml_nelements(t) : 0;
            }
            if (!ok) {
                continue;
            }
            ggml_tensor * consts = ggml_new_tensor_1d(fused_context(), GGML_TYPE_F32, n);
            ggml_format_name(consts, "blk.%d.attn_consts.weight", i);
            fused_tensor f;
            f.t   = consts;
            f.ref = layer.wo;
            size_t offs = 0;
            for (const ggml_tensor * t : pieces) {
                f.pieces.emplace_back(ggml_get_name(t), offs);
                offs += ggml_nbytes(t);
            }
            fused_pending.push_back(std::move(f));
            attn_consts[i] = consts;
        }
    }

    // the constants of the fused router op
    if (fuse_weights) {
        for (int i = 0; i < n_layer_phys(); ++i) {
            const auto & layer = layers[i];
            if (layer.ffn_norm == nullptr || layer.ffn_norm->type != GGML_TYPE_F32 || layer.ffn_gate_inp == nullptr ||
                    (layer.ffn_exp_probs_b != nullptr && layer.ffn_exp_probs_b->type != GGML_TYPE_F32)) {
                continue;
            }
            const int64_t n_expert = layer.ffn_gate_inp->ne[1];
            ggml_tensor * consts = ggml_new_tensor_1d(fused_context(), GGML_TYPE_F32, n_embd + n_expert);
            ggml_format_name(consts, "blk.%d.ffn_consts.weight", i);
            fused_tensor f;
            f.t   = consts;
            f.ref = layer.wo;
            f.pieces.emplace_back(ggml_get_name(layer.ffn_norm), 0);
            if (layer.ffn_exp_probs_b != nullptr) {
                f.pieces.emplace_back(ggml_get_name(layer.ffn_exp_probs_b), n_embd*sizeof(float));
            }
            f.zero_fill = layer.ffn_exp_probs_b == nullptr; // the bias part stays zero
            fused_pending.push_back(std::move(f));
            ffn_consts[i] = consts;
        }
    }

    load_fused(ml);

    // the cache slots past the physical layers share the weights
    for (int il = n_layer_phys(); il < (int) hparams.n_layer(); ++il) {
        for (int ip = 0; ip < n_layer_phys(); ++ip) {
            if (layers[il].wo == layers[ip].wo) {
                layers[il] = layers[ip];
                break;
            }
        }
    }

    LLAMA_LOG_INFO("%s: fused the attention input projections of %d layers\n", __func__, n_fused);
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

void llama_model_inkling2::graph::init_fused() {
    static const bool enabled = inkling_enabled("LLAMA_INKLING_FUSED");
    // the fused ops bypass the LoRA hooks of the controller projections
    fused = enabled && (loras == nullptr || loras->empty());
    moe_sum_mul_mat = fused;

    static const bool q8_enabled = [] {
        const char * env = getenv("LLAMA_INKLING_Q8_ACT");
        return env == nullptr || atoi(env) != 0;
    }();
    q8_act = fused && q8_enabled;

    static const bool attn_op_enabled = inkling_enabled("LLAMA_INKLING_ATTN_OP");
    attn_op = fused && attn_op_enabled;

    // 0: build_moe_ffn; 3 (default): merged gate|up matmul + SITU, down matmul, fused sum + conv + residual;
    // 1: gate/up as two fusable matmuls (same speed); 2: the custom MoE kernels (slower, kept for tuning)
    static const int moe_op_mode = [] {
        const char * env = getenv("LLAMA_INKLING_MOE_OP");
        return env == nullptr ? 3 : atoi(env);
    }();
    moe_op = fused ? moe_op_mode : 0;

    static const bool router_op_enabled = [] {
        const char * env = getenv("LLAMA_INKLING_ROUTER_OP");
        return env == nullptr || atoi(env) != 0;
    }();
    router_op = fused && router_op_enabled;

    // opt-in for now: measured a wash on the RTX 3090 (the glue's dependent round trips inflate ~3.5x under the
    // concurrent weight stream, see perf/RESULTS.md phase 3)
    static const bool proj_op_enabled = [] {
        const char * env = getenv("LLAMA_INKLING_PROJ_OP");
        return env != nullptr && atoi(env) != 0;
    }();
    proj_op = fused && attn_op && proj_op_enabled;

    // windows (speculative verify): every fused op of the single-token path takes the rows of the window
    static const bool mr_enabled = [] {
        const char * env = getenv("IK_MR");
        return env != nullptr && atoi(env) != 0;
    }();
    mr = mr_enabled && fused && q8_act && attn_op && router_op && moe_op == 3 && !proj_op && n_tokens >= 2 && n_tokens <= 8 &&
        ubatch.n_seqs == 1 && ubatch.n_seq_tokens == (uint32_t) n_tokens && cparams.flash_attn && !cparams.moe_random_routing &&
        n_embd == 2304 && hparams.n_shortconv_l_cache == 4;
    if (mr) {
        const auto & m = static_cast<const llama_model_inkling2 &>(model);
        for (const auto & fa : m.fused_attn) {
            // the projections with a row-exact window kernel
            mr = mr && (fa.w == nullptr || fa.w->type == GGML_TYPE_IQ4_NL || fa.w->type == GGML_TYPE_Q6_K);
        }
        for (size_t il = 0; il < model.layers.size() && il < (size_t) hparams.n_layer(); ++il) {
            mr = mr && model.layers[il].attn_post_conv != nullptr && model.layers[il].ffn_post_conv != nullptr;
        }
    }
}

static void ik_set_param(ggml_tensor * t, int i, int32_t v) {
    t->op_params[i] = v;
}

// mark a matmul of a window for the row-exact kernels of the backend (see GGML_IK_MR_MAGIC)
ggml_tensor * llama_model_inkling2::graph::mr_mark(ggml_tensor * t, int q_rows) {
    if (mr && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID)) {
        ik_set_param(t, 13, (int32_t) n_tokens);
        ik_set_param(t, 14, q_rows);
        ik_set_param(t, 15, GGML_IK_MR_MAGIC);
    }
    return t;
}

// the attention-output convolution can be folded into the router op of this layer: single-token decode, the fused
// router with the direct state access, no loop messages (their mixes consume the attention output first), <= 5 states
bool llama_model_inkling2::graph::router_conv_ok(int il, int ip, bool has_messages) const {
    const auto & layer = model.layers[il];
    const ggml_tensor * ffn_consts = static_cast<const llama_model_inkling2 &>(model).ffn_consts[ip];
    const int n_blocks = (int) (ip/hparams.attn_res_block_size) + 1; // the states of the FFN mix: the blocks (incl. this one) + cur
    // opt-in: measured a wash on the RTX 3090 (the 16-block router kernel is limited by its per-SM data transfer)
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_INKLING_ROUTER_CONV");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled && router_op && moe_op == 3 && !has_messages && n_tokens == 1 && ubatch.n_seqs == 1 && rs_ids != nullptr && ffn_consts != nullptr &&
        layer.ffn_gate_inp->type == GGML_TYPE_F32 && hparams.expert_weights_norm_eps > 0.0f && hparams.n_expert % 8 == 0 && hparams.n_expert <= 256 &&
        n_embd % 4 == 0 && n_embd <= 2304 && n_blocks + 1 <= 5 && (2*hparams.n_embd_k_gqa(il)) % 4 == 0 && layer.ffn_post_conv != nullptr &&
        hparams.n_shortconv_l_cache == 4 && layer.attn_post_conv != nullptr &&
        layer.ffn_gate_up_exps != nullptr && hparams.n_expert_used(il) <= 8 && !cparams.moe_random_routing;
}

void llama_model_inkling2::graph::flush_pending() {
    if (pending.d == nullptr) {
        return;
    }
    ggml_tensor * t = ggml_inkling_moe_sum_conv_into(ctx0, pending.d, pending.w, pending.conv_w, pending.state, pending.rs_ids, pending.residual,
            pending.out, pending.write_cell, pending.chan_offset, pending.scale);
    ggml_build_forward_expand(gf, t);
    pending.d = nullptr;
}

// Q8_1 activations go to the quantized matvec kernels of the CUDA and CPU backends
ggml_type llama_model_inkling2::graph::act_type(int il) const {
    // only the matvec path consumes Q8_1 activations: at most 8 columns, and the attention projection of the three
    // inputs of a full-attention or looped layer takes 3*n_tokens columns
    if (!q8_act || (3*n_tokens > 8 && !mr)) {
        return GGML_TYPE_F32;
    }
    ggml_backend_dev_t dev = model.dev_layer(il);
    const char * name = dev ? ggml_backend_dev_name(dev) : "";
    const bool ok = strncmp(name, "CUDA", 4) == 0 || (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU);
    return ok ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
}

// the mixed output of query j as [n_embd, n_tokens]; the fused mix is laid out [n_embd, n_tokens, n_q], the plain one [n_embd, n_q, n_tokens]
ggml_tensor * llama_model_inkling2::graph::mixed_view(ggml_tensor * mixed, int j) {
    if (fused) {
        return ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j*mixed->nb[2]);
    }
    return ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
}

ggml_tensor * llama_model_inkling2::graph::zero_sinks(int64_t n_head) {
    if (!fused) {
        return ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
    }
    if (sinks_zero == nullptr || sinks_n_head != n_head) {
        sinks_zero   = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        sinks_n_head = n_head;
    }
    return sinks_zero;
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (fused) {
        return ggml_l2_clip(ctx0, cur, scale, threshold, 1e-6f);
    }

    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// fused: [n_embd, n_tokens, n_q], plain: [n_embd, n_q, n_tokens] (see mixed_view)
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    if (fused) {
        std::vector<ggml_tensor *> states;
        states.reserve(blocks.size() + 1);
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            states.push_back(ggml_reshape_2d(ctx0, *it, n_embd, n_tokens));
        }
        states.push_back(ggml_reshape_2d(ctx0, cur, n_embd, n_tokens));

        return ggml_res_mix(ctx0, query, states.data(), (int) states.size(), 1e-6f, 1.0f/sqrtf(n_embd));
    }

    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// the fused mix, RMS-normalized and scaled by norm_w: [n_embd, n_tokens, n_q] in out_type; query == nullptr: the norm of cur alone
ggml_tensor * llama_model_inkling2::graph::mix_norm(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query,
        ggml_tensor * norm_w, ggml_type out_type) {
    GGML_ASSERT(fused);
    std::vector<ggml_tensor *> states;
    states.reserve(blocks.size() + 1);
    if (query) {
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            states.push_back(ggml_reshape_2d(ctx0, *it, n_embd, n_tokens));
        }
    }
    states.push_back(ggml_reshape_2d(ctx0, cur, n_embd, n_tokens));

    return ggml_res_mix_norm(ctx0, query, states.data(), (int) states.size(), 1e-6f, 1.0f/sqrtf(n_embd), norm_w, hparams.f_norm_rms_eps, out_type);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
// the optional residual is added to the scaled result
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        ggml_tensor * residual,
        ggml_shortconv_post post,
        int64_t n_heads_post,
        float post_eps,
        float post_threshold) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    const float scale = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);

    // history: [n_conv - 1, channels, n_seqs], or all the cells of the cache when the fused op reads them by rs_ids
    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, history->ne[2], history->nb[1], history->nb[2], offset*history->nb[1]);

    if (fused) {
        // the fused op convolves [past, cur] and stores the new state itself
        // the taps may be a strided view of the fused projection output
        ggml_tensor * taps = dynamic ? ggml_view_3d(ctx0, dynamic, n_conv, dynamic->ne[0]/n_conv, n_tokens, n_conv*sizeof(float), dynamic->nb[1], 0) : nullptr;
        ggml_tensor * state_view = build_conv_state_view(inp, state, n_conv - 1, n_channels, n_seqs, n_seq_tokens,
                offset*(n_conv - 1)*ggml_element_size(state));

        return ggml_shortconv_res_post(ctx0, past, weight, cur, taps, residual, state_view, rs_ids, scale, post, n_heads_post, post_eps, post_threshold);
    }

    GGML_ASSERT(rs_ids == nullptr);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_view_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs, cur->nb[1], n_seq_tokens*cur->nb[1], 0));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_view_3d(ctx0, dynamic, n_conv, n_heads, n_tokens, n_conv*sizeof(float), dynamic->nb[1], 0);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    out = ggml_scale(ctx0, ggml_add(ctx0, cur, out), scale);
    out = residual ? ggml_add(ctx0, residual, out) : out;

    // the per-head post-normalization of the plain graph
    if (post != GGML_SHORTCONV_POST_NONE) {
        const int64_t hd = n_channels/n_heads_post;
        ggml_tensor * heads = ggml_reshape_3d(ctx0, out, hd, n_heads_post, n_tokens);
        if (post == GGML_SHORTCONV_POST_RMS_NORM) {
            heads = ggml_rms_norm(ctx0, heads, post_eps);
        } else {
            heads = clip(heads, nullptr, post_threshold);
        }
        out = ggml_reshape_2d(ctx0, heads, n_channels, n_tokens);
    }

    return out;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// the optional norm weight RMS-normalizes the result (fused into the op)
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl, ggml_tensor * norm_w, int il) {
    if (fused) {
        ggml_tensor * d = ggml_loop_ctrl_down(ctx0, loop.ctrl_down[ctrl], cur, message, 1e-6f);

        return ggml_loop_ctrl_apply(ctx0, cur, message, d, loop.ctrl_gain[ctrl], loop.ctrl_forget[ctrl], norm_w, hparams.f_norm_rms_eps);
    }

    ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    cur = ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));

    return norm_w ? build_norm(cur, norm_w, nullptr, LLM_NORM_RMS, il) : cur;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    if (mr) {
        inp->window_kv_before = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2);
        ggml_set_input(inp->window_kv_before);
        ggml_set_name(inp->window_kv_before, "window_kv_before");
    }
    attn = inp->get_attn();
    recr = inp->get_recr();

    inp_pos     = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](int ctrl) {
        return mixed_view(messages, ctrl);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = nullptr;
    rs_ids = nullptr;
    if (mr) {
        mr_planes      = cparams.n_rs_seq == 0 ? 1 : (int) std::min<int64_t>(n_tokens, (int64_t) cparams.n_rs_seq + 1);
        mr_plane_cells = (int) recr->mctx->get_size();
    }
    if (fused && (ubatch.n_seq_tokens == 1 || mr)) {
        // single-token sequences: the fused convolutions read the cells of the state straight from the cache (by s_copy)
        // and write the new cells; with more tokens the readers and the writers of a channel differ, so the state is gathered
        auto get_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            rs_ids = ids;
            return ggml_view_3d(ctx, states, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, states->ne[1],
                    (hparams.n_shortconv_l_cache - 1)*ggml_element_size(states), states->nb[1], 0);
        };
        history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs, get_rows);
    } else {
        history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
    }

    ggml_tensor * inpSA = cur;

    const auto & fa = static_cast<const llama_model_inkling2 &>(model).fused_attn[ip];
    const bool use_fused = fused && fa.w != nullptr;
    const ggml_type a_type = act_type(il);

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    ggml_tensor * inputs3   = nullptr; // the three inputs as the slabs of one [n_embd, n_tokens, 3] tensor
    ggml_tensor * mixed = nullptr;
    bool inputs_normed = false;
    ggml_tensor * norm3 = is_swa ? layer.attn_norm : static_cast<const llama_model_inkling2 &>(model).attn_inp_norm_qkv[ip];

    // the mix + norm + projection matvec as one persistent kernel (single-token decode), which also finishes the
    // deferred MoE epilogue of the previous layer
    ggml_tensor * proj_pre = nullptr;
    const bool use_proj = proj_op && use_fused && !messages && norm3 != nullptr && a_type == GGML_TYPE_Q8_1 && n_tokens == 1 &&
        ubatch.n_seqs == 1 && rs_ids != nullptr && n_embd % 32 == 0 && n_embd <= 3072 &&
        (fa.w->type == GGML_TYPE_Q8_0 || fa.w->type == GGML_TYPE_Q6_K || fa.w->type == GGML_TYPE_IQ4_NL) &&
        (layer.attn_res_query == nullptr || layer.attn_res_query->ne[1] == (is_swa ? 1 : 3)) && (pending.d == nullptr || pending.out == cur) &&
        blocks.size() + 1 <= 6;
    if (!use_proj) {
        flush_pending();
    }
    if (use_proj) {
        std::vector<ggml_tensor *> states;
        if (layer.attn_res_query) {
            for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
                states.push_back(ggml_reshape_2d(ctx0, *it, n_embd, n_tokens));
            }
        }
        states.push_back(cur);
        const bool epi = pending.d != nullptr;
        proj_pre = ggml_inkling_proj(ctx0, layer.attn_res_query, states.data(), (int) states.size(), norm3, fa.w, is_swa ? 1 : 3, fa.n_q, fa.n_k,
                1e-6f, 1.0f/sqrtf(n_embd), hparams.f_norm_rms_eps,
                epi ? pending.d : nullptr, epi ? pending.w : nullptr, epi ? pending.conv_w : nullptr, epi ? pending.state : nullptr,
                epi ? pending.rs_ids : nullptr, epi ? pending.residual : nullptr, epi ? pending.out : nullptr,
                pending.write_cell, pending.chan_offset, pending.scale);
        cb(proj_pre, "attn_proj", il);
        // the op writes the layer output (the previous epilogue) as a side output: it must precede every other consumer
        ggml_build_forward_expand(gf, proj_pre);
        pending.d = nullptr;
        inputs_normed = true;
    } else if (use_fused && !messages && norm3 != nullptr) {
        // the mix and the input norm in one op, written as the activation of the projection
        ggml_tensor * x = mix_norm(blocks, cur, layer.attn_res_query, norm3, a_type);
        cb(x, "attn_inp", il);
        if (is_swa) {
            inputs[0] = inputs[1] = inputs[2] = ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[1], 0);
        } else {
            inputs3 = x;
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[1], j*x->nb[2]);
            }
        }
        inputs_normed = true;
    } else if (layer.attn_res_query) {
        mixed = mix(blocks, cur, layer.attn_res_query);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = mixed_view(mixed, is_swa ? 0 : j);
        }
    }

    if (messages && fused && loop->ctrl_down_qkv) {
        // the q, k and v controllers in one batched op each, the input norm fused into the second
        ggml_tensor * msgs3 = ggml_view_3d(ctx0, messages, n_embd, n_tokens, 3, messages->nb[1], messages->nb[2], 0);
        ggml_tensor * cur3  = inputs[0];
        ggml_tensor * norm3 = layer.attn_norm;
        if (!is_swa) {
            // the Q, K and V inputs are the three mixes: [n_embd, n_tokens, 3]
            GGML_ASSERT(mixed != nullptr && mixed->ne[2] == 3);
            cur3  = mixed;
            norm3 = static_cast<const llama_model_inkling2 &>(model).attn_inp_norm_qkv[ip];
        }
        if (norm3 != nullptr) {
            ggml_tensor * d    = ggml_loop_ctrl_down(ctx0, loop->ctrl_down_qkv, cur3, msgs3, 1e-6f);
            ggml_tensor * out3 = ggml_loop_ctrl_apply_ext(ctx0, cur3, msgs3, d, loop->ctrl_gain_qkv, loop->ctrl_forget_qkv, norm3, hparams.f_norm_rms_eps,
                    use_fused ? a_type : GGML_TYPE_F32);
            cb(out3, "attn_ctrl", il);
            inputs3 = out3;
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, out3, n_embd, n_tokens, out3->nb[1], j*out3->nb[2]);
            }
            inputs_normed = true;
        }
    }

    if (messages && !inputs_normed) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }

    if (inputs_normed) {
        // normalized by the controllers
    } else if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    ggml_tensor * attn_proj = nullptr; // the fused projection output, [n_q, n_tokens]
    {
        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        ggml_tensor * Vcur = nullptr;
        ggml_tensor * Rcur = nullptr;
        ggml_tensor * k_taps = nullptr;
        ggml_tensor * v_taps = nullptr;
        if (use_fused) {
            // the Q group, and the K and V groups when their inputs differ; else everything in one matmul
            ggml_tensor * out_q = nullptr;
            ggml_tensor * out_k = nullptr;
            ggml_tensor * out_v = nullptr;
            ggml_tensor * proj  = nullptr; // the whole projection output, [n_rows, 1 or 3 columns]
            if (proj_pre != nullptr) {
                proj = proj_pre;
                if (proj->ne[1] == 1) {
                    out_q = proj;
                    out_k = ggml_view_2d(ctx0, proj, fa.n_k, n_tokens, proj->nb[1], fa.n_q*sizeof(float));
                    out_v = ggml_view_2d(ctx0, proj, fa.n_v, n_tokens, proj->nb[1], (fa.n_q + fa.n_k)*sizeof(float));
                } else {
                    out_q = ggml_view_2d(ctx0, proj, fa.n_q, n_tokens, proj->nb[1], 0);
                    out_k = ggml_view_2d(ctx0, proj, fa.n_k, n_tokens, proj->nb[1], 1*proj->nb[1] + fa.n_q*sizeof(float));
                    out_v = ggml_view_2d(ctx0, proj, fa.n_v, n_tokens, proj->nb[1], 2*proj->nb[1] + (fa.n_q + fa.n_k)*sizeof(float));
                }
            } else if (inputs[0] == inputs[1] && inputs[0] == inputs[2]) {
                out_q = mr_mark(ggml_mul_mat(ctx0, fa.w, inputs[0]));
                out_k = ggml_view_2d(ctx0, out_q, fa.n_k, n_tokens, out_q->nb[1], fa.n_q*sizeof(float));
                out_v = ggml_view_2d(ctx0, out_q, fa.n_v, n_tokens, out_q->nb[1], (fa.n_q + fa.n_k)*sizeof(float));
                proj  = out_q;
            } else if (inputs3 != nullptr && (n_tokens <= 4 || mr)) {
                // few tokens: one matmul over the three inputs reads the weights once; the groups take their own columns
                ggml_tensor * x3  = ggml_reshape_2d(ctx0, inputs3, n_embd, 3*n_tokens);
                ggml_tensor * out = mr_mark(ggml_mul_mat(ctx0, fa.w, x3), (int) fa.n_q); // [n_q + n_k + n_v, 3*n_tokens]
                proj  = out;
                out_q = ggml_view_2d(ctx0, out, fa.n_q, n_tokens, out->nb[1], 0);
                out_k = ggml_view_2d(ctx0, out, fa.n_k, n_tokens, out->nb[1], 1*n_tokens*out->nb[1] + fa.n_q*sizeof(float));
                out_v = ggml_view_2d(ctx0, out, fa.n_v, n_tokens, out->nb[1], 2*n_tokens*out->nb[1] + (fa.n_q + fa.n_k)*sizeof(float));
            } else {
                out_q = ggml_mul_mat(ctx0, fa.q, inputs[0]);
                out_k = ggml_mul_mat(ctx0, fa.k, inputs[1]);
                out_v = ggml_mul_mat(ctx0, fa.v, inputs[2]);
            }
            cb(out_q, "attn_proj", il);

            // single-token decode: the whole attention block (K/V convolutions and stores, clips, bias, attention, gate) in one op
            ggml_tensor * consts = static_cast<const llama_model_inkling2 &>(model).attn_consts[ip];
            ggml_tensor * mask_raw = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
            if (attn_op && proj != nullptr && consts != nullptr && rs_ids != nullptr && (n_tokens == 1 || mr) && ubatch.n_seqs == 1 &&
                    mask_raw->ne[3] == 1 && mask_raw->ne[2] == 1 && n_embd_head == 128 && cparams.flash_attn) {
                const auto * mctx_cur = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
                ggml_tensor * k = mctx_cur->get_k(ctx0, il);
                ggml_tensor * v = mctx_cur->get_v(ctx0, il);
                ggml_tensor * kv_idxs = is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs();
                ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;
                const bool kv_ok = k->type == v->type && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && v->nb[1] < v->nb[2] &&
                    k->ne[3] == 1 && rel_pos->ne[0] == k->ne[2] && mask_raw->ne[0] == k->ne[2];
                // An explicitly rotated cache must go through build_attn_masked,
                // which applies the matching transforms to queries and values.
                const bool rotated = is_swa ? (attn->self_k_rot_swa[il] || attn->self_v_rot_swa[il])
                                            : (attn->self_k_rot[il] || attn->self_v_rot[il]);
                if (kv_ok && !rotated) {
                    const auto & hp = hparams;
                    cur = ggml_inkling_attn(ctx0, proj, consts, state, rs_ids, k, v, kv_idxs, rel_pos, mask_raw, is_swa ? nullptr : inp_pos,
                            n_head, n_head_kv, n_embd_head, hp.n_relative, hp.n_shortconv_l_cache, !is_swa, recr->mctx->get_head(),
                            hp.f_qk_clip, hp.f_relative_clip, hp.f_norm_rms_eps, hp.f_log_scale_alpha, (float) hp.n_log_scale_floor,
                            1.0f/sqrtf(n_embd_head), a_type);
                    if (mr) {
                        cur->src[10] = inp->window_kv_before;
                        ik_set_param(cur, 14, mr_planes);
                        ik_set_param(cur, 15, mr_plane_cells);
                    }
                    cb(cur, "attn_gated", il);
                    // the V cell indices are set with the K ones: keep their input allocated
                    ggml_build_forward_expand(gf, is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs());

                    const int ip_block = ip % hparams.attn_res_block_size;
                    ggml_tensor * residual = ip_block == 0 ? nullptr : inpSA;
                    if (router_conv_ok(il, ip, messages != nullptr)) {
                        // the wo conv + residual is computed by the router op (its last state is the attention output)
                        wo_out_pending   = build_lora_mm(layer.wo, cur);
                        wo_residual      = residual;
                        cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);
                        cb(cur, "attn_out_leaf", il);
                    } else {
                        cur = convolve(mr_mark(build_lora_mm(layer.wo, cur)), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa, residual);
                    }
                    cb(cur, "attn_out", il);
                    goto attn_done;
                }
            }
            GGML_ASSERT(!mr && "inkling2: a window needs the fused attention op");

            Qcur = ggml_view_3d(ctx0, out_q, n_embd_head, n_head, n_tokens, n_embd_head*sizeof(float), out_q->nb[1], 0);
            Rcur = ggml_view_3d(ctx0, out_q, hparams.n_relative, n_head, n_tokens, hparams.n_relative*sizeof(float), out_q->nb[1], fa.off_rel*sizeof(float));
            Kcur = ggml_view_2d(ctx0, out_k, n_embd_k_gqa, n_tokens, out_k->nb[1], 0);
            Vcur = ggml_view_2d(ctx0, out_v, n_embd_k_gqa, n_tokens, out_v->nb[1], 0);
            if (!is_swa) {
                const int64_t n_conv = hparams.n_shortconv_l_cache;
                k_taps = ggml_view_2d(ctx0, out_k, n_conv*n_head_kv, n_tokens, out_k->nb[1], fa.off_kproj*sizeof(float));
                v_taps = ggml_view_2d(ctx0, out_v, n_conv*n_head_kv, n_tokens, out_v->nb[1], fa.off_vproj*sizeof(float));
            }
            attn_proj = out_q;
        } else {
            Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, inputs[0]), n_embd_head, n_head, n_tokens);
            Rcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.attn_rel, inputs[0]), hparams.n_relative, n_head, n_tokens);
            Kcur = build_lora_mm(layer.wk, inputs[1]);
            Vcur = build_lora_mm(layer.wv, inputs[2]);
            if (!is_swa) {
                k_taps = build_lora_mm(layer.attn_k_conv_proj, inputs[1]);
                v_taps = build_lora_mm(layer.attn_v_conv_proj, inputs[2]);
            }
        }

        // the per-head clip of K and norm of V are fused into their convolutions
        Kcur = convolve(Kcur, layer.attn_k_conv, k_taps, state, history, recr, 0, nullptr,
                GGML_SHORTCONV_POST_L2_CLIP, n_head_kv, 1e-6f, hparams.f_qk_clip);
        Vcur = convolve(Vcur, layer.attn_v_conv, v_taps, state, history, recr, n_embd_k_gqa, nullptr,
                GGML_SHORTCONV_POST_RMS_NORM, n_head_kv, hparams.f_norm_rms_eps, 0.0f);

        Qcur = clip(Qcur, layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        Rcur = clip(Rcur, layer.attn_rel_clip, hparams.f_relative_clip);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        ggml_tensor * mask_raw = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw);

        const int64_t n_stream = mask_raw->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        ggml_tensor * sinks = use_fused ?
            ggml_view_2d(ctx0, attn_proj, n_head, n_tokens, attn_proj->nb[1], fa.off_sink*sizeof(float)) :
            build_lora_mm(layer.attn_sink_proj, inputs[0]);

        if (fused && n_stream == 1 && mask_raw->ne[2] == 1) {
            // the bias table projected by r, gathered by distance, with the sink logits and the mask: the per-head mask
            ggml_tensor * kq_mask = ggml_rel_bias(ctx0, Rcur, layer.attn_rel_bias, sinks, rel_pos, mask_raw,
                    cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(kq_mask, "kq_b", il);

            cur = build_attn_masked(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_mask, zero_sinks(n_head), 1.0f/sqrtf(n_embd_head), il);
        } else {
            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

            // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, ggml_cont(ctx0, sinks), 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
            cb(kq_b, "kq_b", il);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, zero_sinks(n_head), 1.0f/sqrtf(n_embd_head), il);
        }

        ggml_tensor * gate = use_fused ?
            ggml_view_2d(ctx0, attn_proj, n_embd_head*n_head, n_tokens, attn_proj->nb[1], fa.off_gate*sizeof(float)) :
            build_lora_mm(layer.wqkv_gate, inputs[0]);
        if (fused) {
            cur = ggml_siglu_split(ctx0, gate, cur);
        } else {
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        }

        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        ggml_tensor * residual = ip % hparams.attn_res_block_size == 0 ? nullptr : inpSA;

        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa, residual);
        cb(cur, "attn_out", il);
    }
attn_done:

    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    const int64_t n_expert      = hparams.n_expert;
    const int64_t n_expert_used = hparams.n_expert_used(il);
    const ggml_tensor * gate_up = layer.ffn_gate_up_exps;
    const ggml_tensor * down = layer.ffn_down_exps;
    const bool ordinary_experts = gate_up && down && gate_up->type == down->type &&
        (gate_up->type == GGML_TYPE_IQ4_NL || gate_up->type == GGML_TYPE_Q8_0);
    // Mode 3 dispatches each expert matrix separately. Its Q8_1 activation
    // requires a supported MMVQ for both weights on the same CUDA device.
    bool mixed_cuda_experts = false;
    if (moe_op == 3 && (n_tokens == 1 || mr) && a_type == GGML_TYPE_Q8_1 && gate_up && down &&
            (gate_up->type == GGML_TYPE_Q2_K || gate_up->type == GGML_TYPE_Q3_K) &&
            (down->type == GGML_TYPE_Q2_0 || down->type == GGML_TYPE_Q4_0 || down->type == GGML_TYPE_IQ4_NL)) {
        const auto dev = model.dev_layer(il);
        if (dev && strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0 && gate_up->buffer && down->buffer) {
            const auto buft = ggml_backend_dev_buffer_type(dev);
            mixed_cuda_experts = ggml_backend_buffer_get_type(gate_up->buffer) == buft &&
                                 ggml_backend_buffer_get_type(down->buffer) == buft;
        }
    }
    const bool moe_fused = moe_op && (n_tokens == 1 || mr) && ubatch.n_seqs == 1 && rs_ids != nullptr &&
        !layer.ffn_up_exps_s && !layer.ffn_gate_exps_s && !layer.ffn_down_exps_s && n_expert_used <= 8 && !cparams.moe_random_routing &&
        (ordinary_experts || mixed_cuda_experts) && n_embd <= 4096;
    // the fused router op: the mix (or the controlled input), the FFN norm, the router and the top-k selection in one op
    ggml_tensor * ffn_consts = static_cast<const llama_model_inkling2 &>(model).ffn_consts[ip];
    const bool router_fused = moe_fused && router_op && ffn_consts != nullptr && n_expert % 8 == 0 && n_expert <= 256 && n_embd % 4 == 0 &&
        layer.ffn_gate_inp->type == GGML_TYPE_F32 && hparams.expert_weights_norm_eps > 0.0f;

    ggml_tensor * selected_experts = nullptr; // [n_expert_used, 1] I32
    ggml_tensor * weights          = nullptr; // [1, n_expert_used, 1] F32 (or the I32 bits of the fused router)
    ggml_tensor * x_q8             = nullptr; // the FFN input as the Q8_1 activation of the expert matmul
    if (router_fused) {
        ggml_tensor * route = mr ? ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 2*n_expert_used, n_tokens) : ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2*n_expert_used);
        cb(route, "ffn_moe_route", il);
        if (a_type == GGML_TYPE_Q8_1) {
            x_q8 = mr ? ggml_new_tensor_2d(ctx0, GGML_TYPE_Q8_1, n_embd, n_tokens) : ggml_new_tensor_1d(ctx0, GGML_TYPE_Q8_1, n_embd);
            cb(x_q8, "ffn_norm_q8", il);
        }
        std::vector<ggml_tensor *> states;
        ggml_tensor * query = nullptr;
        if (messages) {
            // the controlled mix: its norm is in the router op
            ggml_tensor * mixed_ffn = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
            states.push_back(control(mixed_ffn, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN));
        } else {
            query = layer.ffn_res_query;
            for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
                states.push_back(ggml_reshape_2d(ctx0, *it, n_embd, n_tokens));
            }
            // the leaf itself (not a view) when it is the attention-output leaf the router op writes
            states.push_back(cur->ne[1] == n_tokens && ggml_is_contiguous(cur) ? cur : ggml_reshape_2d(ctx0, cur, n_embd, n_tokens));
        }
        if (wo_out_pending) {
            GGML_ASSERT(states.back() == ffn_inp);
            cur = ggml_inkling_router_conv(ctx0, query, states.data(), (int) states.size(), ffn_consts, layer.ffn_gate_inp, route, x_q8,
                    wo_out_pending, layer.attn_post_conv, state, rs_ids, wo_residual, recr->mctx->get_head(), 2*n_embd_k_gqa,
                    1.0f/sqrtf(hparams.n_shortconv_l_cache + 1),
                    1e-6f, 1.0f/sqrtf(n_embd), hparams.f_norm_rms_eps, n_expert_used, hparams.expert_weights_scale, hparams.expert_weights_norm_eps);
            wo_out_pending = nullptr;
        } else {
            cur = ggml_inkling_router(ctx0, query, states.data(), (int) states.size(), ffn_consts, layer.ffn_gate_inp, route, x_q8,
                    1e-6f, 1.0f/sqrtf(n_embd), hparams.f_norm_rms_eps, n_expert_used, hparams.expert_weights_scale, hparams.expert_weights_norm_eps);
        }
        cb(cur, "ffn_norm", il);
        // the op writes route and x_q8 as side outputs: it must be in the graph even when its F32 result is not consumed
        ggml_build_forward_expand(gf, cur);
        if (mr) {
            selected_experts = ggml_view_2d(ctx0, route, n_expert_used, n_tokens, route->nb[1], 0);
            weights          = ggml_view_3d(ctx0, route, 1, n_expert_used, n_tokens, route->nb[0], route->nb[1], route->nb[0]*n_expert_used);
        } else {
        selected_experts = ggml_view_2d(ctx0, route, n_expert_used, 1, route->nb[0]*n_expert_used, 0);
        weights          = ggml_view_3d(ctx0, route, 1, n_expert_used, 1, route->nb[0], route->nb[0]*n_expert_used, route->nb[0]*n_expert_used);
        }
    } else if (fused && !messages) {
        // the mix and the FFN norm in one op (F32: the router reads it too)
        cur = ggml_reshape_2d(ctx0, mix_norm(blocks, cur, layer.ffn_res_query, layer.ffn_norm, GGML_TYPE_F32), n_embd, n_tokens);
        cb(cur, "ffn_norm", il);
    } else {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
        if (messages) {
            cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
        }
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    }

    if (moe_fused && !router_fused) {
        // the router as in build_moe_ffn (sigmoid gating, normalized weights, scale), then the experts in two ops
        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur); // [n_expert, 1]
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        cb(logits, "ffn_moe_logits", il);
        ggml_tensor * probs = ggml_sigmoid(ctx0, logits);
        cb(probs, "ffn_moe_probs", il);
        ggml_tensor * selection_probs = probs;
        if (layer.ffn_exp_probs_b != nullptr) {
            selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
            cb(selection_probs, "ffn_moe_probs_biased", il);
        }
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, 1]
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
        weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, 1]
        cb(weights, "ffn_moe_weights", il);
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        cb(weights_sum, "ffn_moe_weights_sum", il);
        if (hparams.expert_weights_norm_eps > 0.0f) {
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0f, hparams.expert_weights_norm_eps);
        } else {
            weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        }
        weights = ggml_div(ctx0, weights, weights_sum);
        cb(weights, "ffn_moe_weights_norm", il);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }
        ggml_build_forward_expand(gf, weights);
    }
    GGML_ASSERT(!mr || (moe_fused && router_fused && x_q8 != nullptr));
    if (moe_fused) {

        const float conv_scale = 1.0f/sqrtf(hparams.n_shortconv_l_cache + 1);
        if (moe_op == 2 && !router_fused) {
            ggml_tensor * h = ggml_inkling_moe_up(ctx0, layer.ffn_gate_up_exps, cur, selected_experts, hparams.situ_beta, hparams.situ_linear_beta);
            cb(h, "ffn_moe_situ", il);
            cur = ggml_inkling_moe_down(ctx0, layer.ffn_down_exps, h, selected_experts, weights, layer.ffn_post_conv, state, rs_ids, ffn_inp,
                    recr->mctx->get_head(), 2*n_embd_k_gqa + n_embd, conv_scale);
        } else {
            ggml_tensor * gu   = layer.ffn_gate_up_exps;
            const int64_t n_ff = gu->ne[1]/2;
            // the expert activation: the router's quantized output when there is one
            ggml_tensor * x3   = x_q8 ? ggml_reshape_3d(ctx0, x_q8, n_embd, 1, mr ? n_tokens : 1) : ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
            ggml_tensor * h    = nullptr;
            if (moe_op == 3) {
                // the merged gate|up matmul, then the activation on its two halves, quantized for the down matmul
                ggml_tensor * gate_up = mr_mark(ggml_mul_mat_id(ctx0, gu, x3, selected_experts)); // [2*n_ff, n_expert_used, 1]
                ggml_tensor * gate = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                ggml_tensor * up   = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff*gate_up->nb[0]);
                h = ggml_situ_split_out(ctx0, gate, up, hparams.situ_beta, hparams.situ_linear_beta, n_ff % ggml_blck_size(GGML_TYPE_Q8_1) == 0 ? a_type : GGML_TYPE_F32);
            } else {
                // the gate and up halves of the merged matrix as two matmuls with the activation: one fused matvec launch (CUDA)
                ggml_tensor * gate_w = ggml_view_3d(ctx0, gu, n_embd, n_ff, n_expert, gu->nb[1], gu->nb[2], 0);
                ggml_tensor * up_w   = ggml_view_3d(ctx0, gu, n_embd, n_ff, n_expert, gu->nb[1], gu->nb[2], n_ff*gu->nb[1]);
                // the views and the reshape enter the graph first, so the two matmuls and the GLU are consecutive nodes (CUDA fuses them)
                ggml_build_forward_expand(gf, gate_w);
                ggml_build_forward_expand(gf, up_w);
                ggml_build_forward_expand(gf, x3);
                ggml_tensor * gate = ggml_mul_mat_id(ctx0, gate_w, x3, selected_experts); // [n_ff, n_expert_used, 1]
                ggml_tensor * up   = ggml_mul_mat_id(ctx0, up_w,   x3, selected_experts);
                h = ggml_situ_split(ctx0, gate, up, hparams.situ_beta, hparams.situ_linear_beta);
            }
            cb(h, "ffn_moe_situ", il);
            ggml_tensor * d = mr_mark(ggml_mul_mat_id(ctx0, layer.ffn_down_exps, h, selected_experts)); // [n_embd, n_expert_used, 1]
            cb(d, "ffn_moe_down", il);
            if (proj_op && router_fused) {
                // the epilogue is deferred into the next layer's fused projection op (or flushed before any other consumer)
                GGML_ASSERT(pending.d == nullptr);
                ggml_tensor * out = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);
                cb(out, "l_out_leaf", il);
                pending.d = d; pending.w = weights; pending.conv_w = layer.ffn_post_conv; pending.state = state; pending.rs_ids = rs_ids;
                pending.residual = ffn_inp; pending.out = out; pending.write_cell = recr->mctx->get_head();
                pending.chan_offset = 2*n_embd_k_gqa + n_embd; pending.scale = conv_scale;
                cur = out;
            } else {
                cur = ggml_inkling_moe_sum_conv(ctx0, d, weights, layer.ffn_post_conv, state, rs_ids, ffn_inp,
                        recr->mctx->get_head(), 2*n_embd_k_gqa + n_embd, conv_scale);
                if (mr) {
                    ik_set_param(cur, 3, mr_planes);
                    ik_set_param(cur, 4, mr_plane_cells);
                }
            }
        }
        cb(cur, "l_out", il);
    } else {
        cur = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                nullptr,
                nullptr,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used(il),
                LLM_FFN_SITU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                nullptr,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);

        cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd, ffn_inp);
        cb(cur, "l_out", il);
    }

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    flush_pending();
    if (fused) {
        cur = ggml_reshape_2d(ctx0, mix_norm(blocks, cur, model.output_res_query, model.output_norm, GGML_TYPE_F32), n_embd, n_tokens);
    } else {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    }
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    const auto & m = static_cast<const llama_model_inkling2 &>(model);
    auto project = [&](ggml_tensor * w) {
        auto * logits = build_lora_mm(w, cur);
        if (logits->ne[1] >= 2 && logits->ne[1] <= 8) {
            mr_mark(logits);
        }
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        return logits;
    };
    static const bool native = getenv("IK_PEER_NATIVE") && atoi(getenv("IK_PEER_NATIVE")) != 0;
    if (native && m.output_half[1] && fused && (cur->ne[1] == 1 || mr)) {
        cur = project(model.output);
        cur->extra = m.output_half[1];
    } else if (!native && m.output_half[1] && fused && cur->ne[1] <= 8) {
        auto * upper = project(m.output_half[1]);
        ggml_build_forward_expand(gf, upper);
        auto * lower = project(m.output_half[0]);
        cur = ggml_concat(ctx0, lower, upper, 0);
    } else {
        cur = project(model.output);
    }

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // debugging: IK_DBG_TAP=<tensor name> (e.g. "l_out-7", "attn_out-20") returns that tensor, zero-padded to the
    // vocabulary, as the logits; the graph ends there (perf/p6/mr_check.cpp compares windows against single tokens with it)
    if (const char * tap = getenv("IK_DBG_TAP")) {
        ggml_build_forward_expand(gf, cur);
        ggml_tensor * t = ggml_graph_get_tensor(gf, tap);
        if (t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] <= cur->ne[0] && t->ne[1] == cur->ne[1]) {
            t = ggml_cont(ctx0, ggml_reshape_2d(ctx0, t, t->ne[0], t->ne[1]));
            res->t_logits = t->ne[0] < cur->ne[0] ? ggml_pad(ctx0, t, (int) (cur->ne[0] - t->ne[0]), 0, 0, 0) : t;
        }
    }
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    init_fused();

    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
