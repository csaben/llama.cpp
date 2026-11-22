// Write full FP32 vocabulary rows for the Phase 7 single-token comparison.
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

static bool decode(llama_context * ctx, const llama_token * tokens, int n, int pos) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: logits model.gguf corpus.txt\n");
        return 1;
    }
    constexpr int prefix = 200;
    constexpr int steps = 64;
    std::ifstream file(argv[2]);
    std::string text(std::istreambuf_iterator<char>(file), {});
    if (text.empty() || text.size() > std::numeric_limits<int>::max() - 8) {
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens(text.size() + 8);
    const int n_tokens = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, false);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq = 8;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    bool ok = ctx && n_tokens >= prefix + steps && decode(ctx, tokens.data(), prefix, 0);
    for (int pos = prefix; ok && pos < prefix + steps; ++pos) {
        ok = decode(ctx, tokens.data() + pos, 1, pos) &&
             fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), n_vocab, stdout) == size_t(n_vocab);
    }
    if (ctx) {
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return ok && fflush(stdout) == 0 ? 0 : 1;
}
