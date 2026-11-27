// Capture one layer's expert inputs from a 512-token prompt with the p9 evaluator.
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct capture {
    std::string directory;
    std::string prefix;
    bool gate = false;
    bool down = false;
    int window = 0;
};

static void save(const ggml_tensor * t, const std::string & path) {
    GGML_ASSERT(t->ne[3] == 1 && t->nb[0] == ggml_type_size(t->type));
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::ofstream f(path, std::ios::binary);
    for (int64_t k = 0; k < t->ne[2]; ++k) {
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            GGML_ASSERT(f.write(bytes.data() + k*t->nb[2] + j*t->nb[1], row_size));
        }
    }
    f.close();
    GGML_ASSERT(f);
}

static bool collect(ggml_tensor * t, bool ask, void * opaque) {
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    auto & data = *static_cast<capture *>(opaque);
    if (data.window != 0 && (t->op_params[15] != GGML_IK_MR_MAGIC || t->op_params[13] != data.window)) { return false; }
    const bool gate = std::strstr(t->src[0]->name, (data.prefix + ".ffn_gate_up_exps.weight").c_str()) != nullptr;
    const bool down = std::strstr(t->src[0]->name, (data.prefix + ".ffn_down_exps.weight").c_str()) != nullptr;
    if ((!gate || data.gate) && (!down || data.down)) {
        return false;
    }
    if (ask) {
        return true;
    }
    GGML_ASSERT(t->src[1]->type == (data.window ? GGML_TYPE_Q8_1 : GGML_TYPE_F32) && t->src[2]->type == GGML_TYPE_I32 && t->type == GGML_TYPE_F32);
    const std::string prefix = data.directory + (gate ? "/gate" : "/down");
    save(t->src[1], prefix + ".input");
    save(t->src[2], prefix + ".ids");
    save(t, prefix + ".output");
    printf("%s %s weights %s activations %s result %s input_type %s columns %lld\n", t->src[0]->name, ggml_op_name(t->op),
            ggml_backend_buffer_name(t->src[0]->buffer), ggml_backend_buffer_name(t->src[1]->buffer), ggml_backend_buffer_name(t->buffer), ggml_type_name(t->src[1]->type), (long long) t->src[1]->ne[2]);
    if (gate) { data.gate = true; }
    if (down) { data.down = true; }
    return true;
}

static bool capture_windows(llama_model * model, const char * token_file, const std::string & directory, capture & data) {
    std::ifstream input(token_file);
    std::vector<llama_token> tokens;
    llama_token token;
    while (input >> token) { tokens.push_back(token); }
    constexpr int prompt = 1025;
    if (tokens.size() < prompt + 27) { return false; }
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_rs_seq = 6;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = collect;
    cp.cb_eval_user_data = &data;
    data.window = -1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return false; }
    bool ok = llama_decode(ctx, llama_batch_get_one(tokens.data(), prompt)) == 0;
    int pos = prompt;
    for (int n = 2; n <= 7 && ok; ++n) {
        data.window = n;
        data.directory = directory + "/" + std::to_string(n);
        std::filesystem::create_directories(data.directory);
        data.gate = data.down = false;
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int r = 0; r < n; ++r) {
            batch.token[r] = tokens[pos+r];
            batch.pos[r] = pos+r;
            batch.n_seq_id[r] = 1;
            batch.seq_id[r][0] = 0;
            batch.logits[r] = 1;
        }
        batch.n_tokens = n;
        ok = llama_decode(ctx, batch) == 0;
        llama_synchronize(ctx);
        llama_batch_free(batch);
        ok = ok && data.gate && data.down;
        pos += n;
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: capture MODEL CORPUS DIRECTORY\n");
        return 2;
    }
    std::ifstream file(argv[2]);
    std::string text(std::istreambuf_iterator<char>(file), {});
    if (text.empty()) { return 2; }
    llama_backend_init();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    GGML_ASSERT(dev);
    const int layer = std::getenv("P11_LAYER") ? std::atoi(std::getenv("P11_LAYER")) : 0;
    const std::string pattern = "blk[.]" + std::to_string(layer) + "[.].*";
    llama_model_tensor_buft_override locations[] = {{pattern.c_str(), ggml_backend_dev_buffer_type(dev)}, {nullptr, nullptr}};
    auto mp = llama_model_default_params();
    const bool windows = std::getenv("P11_CAPTURE_WINDOWS") != nullptr;
    mp.n_gpu_layers = windows ? 99 : 0;
    if (!windows) { mp.tensor_buft_overrides = locations; }
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    capture result{argv[3], "blk." + std::to_string(layer)};
    if (windows) {
        const bool ok = capture_windows(model, argv[2], argv[3], result);
        llama_model_free(model);
        llama_backend_free();
        return ok ? 0 : 1;
    }
    std::vector<llama_token> tokens(text.size() + 8);
    int count = llama_tokenize(llama_model_get_vocab(model), text.data(), text.size(), tokens.data(), tokens.size(), true, false);
    if (count < 512) {
        fprintf(stderr, "capture requires at least 512 tokens, got %d\n", count);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 10;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = collect;
    cp.cb_eval_user_data = &result;
    llama_context * ctx = llama_init_from_model(model, cp);
    const bool ok = ctx && llama_decode(ctx, llama_batch_get_one(tokens.data(), 512)) == 0;
    if (ctx) { llama_free(ctx); }
    llama_model_free(model);
    llama_backend_free();
    return ok && result.gate && result.down ? 0 : 1;
}
