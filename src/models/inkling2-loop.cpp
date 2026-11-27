#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);

        // the q, k and v controllers are stacked into batched tensors after loading (one op for the three), when F32
        bool stack = fuse_weights;
        for (int c = LOOP_CTRL_Q; stack && c <= LOOP_CTRL_V; ++c) {
            for (const llm_tensor * group : { down, gain, forget }) {
                const ggml_tensor * meta = ml.get_tensor_meta(tn(group[c], "weight", il).str().c_str());
                stack = stack && meta != nullptr && meta->type == GGML_TYPE_F32;
            }
        }
        const int skip = stack ? TENSOR_SKIP | TENSOR_SKIP_QUIET : 0;

        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            const int flags = c <= LOOP_CTRL_V ? skip : 0;
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, flags);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, flags);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, flags);
        }

        // LLAMA_INKLING_CTRL_F16=1: the controller weights are kept as F16 (half the bytes; a quality-gated variant)
        static const bool ctrl_f16 = [] {
            const char * env = getenv("LLAMA_INKLING_CTRL_F16");
            return env != nullptr && atoi(env) != 0;
        }();
        const ggml_type ctrl_type = ctrl_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

        if (stack) {
            ggml_context * ctx = fused_context();
            l.ctrl_down_qkv   = ggml_new_tensor_3d(ctx, ctrl_type, 2*n_embd, rank, 3);
            l.ctrl_gain_qkv   = ggml_new_tensor_3d(ctx, ctrl_type, rank, n_embd, 3);
            l.ctrl_forget_qkv = ggml_new_tensor_3d(ctx, ctrl_type, rank, n_embd, 3);
            ggml_format_name(l.ctrl_down_qkv,   "blk.%u.loop_qkv_ctrl_down.weight",   il);
            ggml_format_name(l.ctrl_gain_qkv,   "blk.%u.loop_qkv_ctrl_gain.weight",   il);
            ggml_format_name(l.ctrl_forget_qkv, "blk.%u.loop_qkv_ctrl_forget.weight", il);

            const struct { ggml_tensor * t; const llm_tensor * group; ggml_tensor ** orig; } stacks[3] = {
                { l.ctrl_down_qkv,   down,   l.ctrl_down   },
                { l.ctrl_gain_qkv,   gain,   l.ctrl_gain   },
                { l.ctrl_forget_qkv, forget, l.ctrl_forget },
            };
            for (const auto & st : stacks) {
                fused_tensor f;
                f.t   = st.t;
                f.ref = layers[il].wo;
                const size_t slab = st.t->nb[2];
                for (int c = LOOP_CTRL_Q; c <= LOOP_CTRL_V; ++c) {
                    const std::string name = tn(st.group[c], "weight", il).str();
                    f.pieces.emplace_back(name, c*slab);
                    // the single controller tensor becomes a view of its slab
                    st.orig[c] = ggml_view_2d(ctx, st.t, st.t->ne[0], st.t->ne[1], st.t->nb[1], c*slab);
                    ggml_set_name(st.orig[c], name.c_str());
                    tensors_by_name.emplace_back(name, st.orig[c]);
                }
                fused_pending.push_back(std::move(f));
            }

            if (ctrl_f16) {
                // the FFN controller too: F16 copies of its three tensors
                ggml_tensor ** targets[3] = { &l.ctrl_down[LOOP_CTRL_FFN], &l.ctrl_gain[LOOP_CTRL_FFN], &l.ctrl_forget[LOOP_CTRL_FFN] };
                for (ggml_tensor ** tp : targets) {
                    ggml_tensor * src = *tp;
                    if (src == nullptr || src->type != GGML_TYPE_F32) {
                        continue;
                    }
                    ggml_tensor * t16 = ggml_new_tensor(ctx, GGML_TYPE_F16, GGML_MAX_DIMS, src->ne);
                    ggml_format_name(t16, "%s.f16", ggml_get_name(src));
                    fused_tensor f;
                    f.t   = t16;
                    f.ref = layers[il].wo;
                    f.pieces.emplace_back(ggml_get_name(src), 0);
                    fused_pending.push_back(std::move(f));
                    *tp = t16;
                }
            }
        }
    }

    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

void llama_model_inkling2_loop::load_arch_tensors_post(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors_post(ml);
    // Every layer of a repeated pass queries the same immutable previous-pass bank.
    // Keep the original matrices for batched prefill and the unfused graph.
    const char * group = getenv("IK_BATCH_MESSAGES");
    if (fuse_weights && !loop_layers.empty() && group && atoi(group) != 0) {
        ggml_tensor * first = loop_layers.front().msg_query;
        bool compatible = first->type == GGML_TYPE_F32 && first->buffer;
        for (const auto & layer : loop_layers) {
            compatible = compatible && layer.msg_query->type == first->type && layer.msg_query->buffer &&
                    ggml_backend_buffer_get_type(layer.msg_query->buffer) == ggml_backend_buffer_get_type(first->buffer);
        }
        if (compatible) {
            core_msg_queries = ggml_new_tensor_2d(fused_context(), first->type, first->ne[0], LOOP_CTRL_COUNT*loop_layers.size());
            ggml_set_name(core_msg_queries, "loop_msg_queries.weight");
            fused_tensor f;
            f.t = core_msg_queries;
            f.ref = first;
            for (size_t i = 0; i < loop_layers.size(); ++i) {
                f.pieces.emplace_back(ggml_get_name(loop_layers[i].msg_query), i*ggml_nbytes(first));
            }
            fused_pending.push_back(std::move(f));
        }
    }
    load_fused(ml); // the controller stacks registered in load_arch_tensors (if not yet loaded)
    if (const char * name = getenv("IK_PEER_CTRL")) {
        auto * dev = ggml_backend_dev_by_name(name);
        if (!dev || strncmp(ggml_backend_dev_name(dev), "CUDA", 4) != 0) {
            throw std::runtime_error("IK_PEER_CTRL requires a CUDA peer");
        }
        if (!core_msg_queries || gate_rank != 64 || hparams.n_embd != 2304 || loop_layers.size() != 12) { return; }
        for (auto & l : loop_layers) {
            if (!l.ctrl_down_qkv || !l.ctrl_down[LOOP_CTRL_FFN] || l.ctrl_down_qkv->type != GGML_TYPE_F16 ||
                    l.ctrl_down[LOOP_CTRL_FFN]->type != GGML_TYPE_F16) { return; }
        }
        auto * buft = ggml_backend_dev_buffer_type(dev);
        peer_msg_down = ggml_new_tensor_3d(fused_context(), GGML_TYPE_F16, 2304, 64, 48);
        ggml_set_name(peer_msg_down, "loop.peer.down.weight");
        ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, peer_msg_down)) };
        if (!buf || ggml_backend_tensor_alloc(buf.get(), peer_msg_down, ggml_backend_buffer_get_base(buf.get())) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("inkling2: failed to allocate peer controller weights");
        }
        std::vector<uint16_t> full(4608*64), half(2304*64);
        for (int i = 0; i < 12; ++i) {
            for (int c = 0; c < 4; ++c) {
                auto * w = c == LOOP_CTRL_FFN ? loop_layers[i].ctrl_down[c] : loop_layers[i].ctrl_down_qkv;
                ggml_backend_tensor_get(w, full.data(), c == LOOP_CTRL_FFN ? 0 : c*w->nb[2], full.size()*sizeof(uint16_t));
                for (int k = 0; k < 64; ++k) {
                    std::copy_n(full.data() + 4608*k + 2304, 2304, half.data() + 2304*k);
                }
                ggml_backend_tensor_set(peer_msg_down, half.data(), (4*i+c)*peer_msg_down->nb[2], half.size()*sizeof(uint16_t));
            }
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        fused_bufs.emplace_back(std::move(buf));
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    int visit = 0;
    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        ggml_tensor * pass_messages = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
            if (fused && (n_tokens == 1 || mr) && m.core_msg_queries) {
                flush_pending();
                pass_messages = mix(bank, bank_out, m.core_msg_queries);
                if (m.peer_msg_down && n_tokens == 1) {
                    ggml_set_name(pass_messages, "loop.peer.messages");
                    pass_messages->extra = m.peer_msg_down;
                }
            }
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            flush_pending(); // bank_out may be a deferred layer output
            ggml_tensor * messages = pass_messages ?
                    ggml_view_3d(ctx0, pass_messages, n_embd, n_tokens, LOOP_CTRL_COUNT, pass_messages->nb[1],
                            pass_messages->nb[2], (ip - start)*LOOP_CTRL_COUNT*pass_messages->nb[2]) :
                    mix(bank, bank_out, loop.msg_query);
            cur = build_layer(il, ip, cur, blocks, messages, &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
