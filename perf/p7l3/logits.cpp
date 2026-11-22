// Write full FP32 vocabulary rows for the Phase 7 single-token comparison.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

static bool decode(llama_context * ctx, const llama_token * tokens, int n, int pos, int seq = 0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

int main(int argc, char ** argv) {
    const bool mixed = argc == 6 && std::string(argv[5]) == "interleave-mixed";
    const bool interleave = mixed || (argc == 6 && std::string(argv[5]) == "interleave");
    if (argc < 3 || argc > 6 || (argc == 6 && !interleave)) {
        fprintf(stderr, "usage: logits model.gguf corpus.txt [prefix [rows [interleave|interleave-mixed]]]\n");
        return 1;
    }
    const int prefix = argc > 3 ? atoi(argv[3]) : 200;
    const int steps = argc > 4 ? atoi(argv[4]) : 64;
    if (prefix <= 0 || steps <= 0 || prefix > 65536 - steps) {
        return 1;
    }
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
    int n_tokens = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, false);
    if (n_tokens > 0 && n_tokens < prefix + steps) {
        const auto corpus = tokens;
        const int corpus_size = n_tokens;
        tokens.resize(prefix + steps);
        for (; n_tokens < prefix + steps; ++n_tokens) {
            tokens[n_tokens] = corpus[n_tokens%corpus_size];
        }
    }
    llama_context_params cp = llama_context_default_params();
    const int sequences = interleave ? 2 : 1;
    cp.n_ctx = std::max(8192, sequences*(prefix + steps));
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = sequences;
    if (interleave) {
        cp.kv_unified = true;
    }
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq = 8;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    bool ok = ctx && n_tokens >= prefix + steps;
    std::vector<llama_token> second;
    if (interleave && ok) {
        second.assign(tokens.begin(), tokens.begin() + prefix + steps);
        std::rotate(second.begin(), second.begin() + second.size()/3, second.end());
    }
    const int widths[] = {7, 15, 31, 47, 127, 129, 512};
    for (int pos = 0, i = 0; ok && pos < prefix; ++i) {
        const int chunk = mixed ? widths[i % std::size(widths)] : (interleave ? 47 : cp.n_batch);
        for (int seq = 0; ok && seq < sequences; ++seq) {
            const auto & input = seq ? second : tokens;
            ok = decode(ctx, input.data() + pos, std::min(chunk, prefix - pos), pos, seq);
        }
        pos += chunk;
    }
    for (int pos = prefix; ok && pos < prefix + steps; ++pos) {
        for (int seq = 0; ok && seq < sequences; ++seq) {
            const auto & input = seq ? second : tokens;
            ok = decode(ctx, input.data() + pos, 1, pos, seq) &&
                 fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), n_vocab, stdout) == size_t(n_vocab);
        }
    }
    if (ctx) {
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return ok && fflush(stdout) == 0 ? 0 : 1;
}
