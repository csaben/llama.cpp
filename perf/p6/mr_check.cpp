// Window (IK_MR) check: the logits of tokens decoded as windows of 2..8 rows must equal, bit for bit, the logits of the
// same tokens decoded one at a time. Two contexts on the same model: A decodes the prefix and then single tokens, B the
// prefix and then windows; B also rolls a window back in part (the speculative-reject path) and redoes the tokens.
// build: g++ -O2 -std=c++17 perf/p6/mr_check.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o build/bin/mr_check
// run:   . perf/p5env.sh; IK_MR=1 LD_LIBRARY_PATH=build/bin build/bin/mr_check $Q2 perf/ppl.txt [prefix_tokens=300] [n_windows=24]
#include "llama.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 2048; cp.n_ubatch = getenv("MR_UBATCH") ? atoi(getenv("MR_UBATCH")) : 512; cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq = 8;
    cp.no_perf = true;
    return llama_init_from_model(model, cp);
}

static void decode(llama_context * ctx, const llama_token * toks, int n, int pos0, bool all_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = all_logits || i == n - 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    llama_batch_free(b);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mr_check model text [prefix] [windows]\n"); return 1; }
    const int n_prefix  = argc > 3 ? atoi(argv[3]) : 300;
    const int n_windows = argc > 4 ? atoi(argv[4]) : 24;
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::stringstream ss; ss << std::ifstream(argv[2]).rdbuf();
    const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 8);
    toks.resize(llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, false));

    // the windows: sizes 2..8 in a fixed mix; after some, roll back the last j rows and redo them
    struct win { int n, rollback; };
    std::vector<win> wins;
    // MR_SIZES="8,3,5" overrides the mix, MR_ROLLBACK=0 disables the partial rollbacks
    std::vector<int> sizes = {4, 2, 8, 3, 5, 7, 6, 4, 8, 2, 3, 8};
    if (const char * e = getenv("MR_SIZES")) {
        sizes.clear();
        for (const char * q = e; *q; ) { sizes.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    }
    const bool rollbacks = !getenv("MR_ROLLBACK") || atoi(getenv("MR_ROLLBACK")) != 0;
    for (int i = 0; i < n_windows; ++i) {
        const int n = sizes[i % sizes.size()];
        wins.push_back({n, rollbacks && i % 3 == 1 ? 1 + (i % (n - 1 > 0 ? n - 1 : 1)) : 0});
    }
    int n_total = 0;
    for (auto & w : wins) n_total += w.n - w.rollback;
    if ((int) toks.size() < n_prefix + n_total + 16) { fprintf(stderr, "text too short\n"); return 1; }

    // A: single tokens
    std::vector<float> ref((size_t) n_total*n_vocab);
    {
        llama_context * ctx = make_ctx(model);
        decode(ctx, toks.data(), n_prefix, 0, false);
        double ms1 = 0.0;
        for (int i = 0; i < n_total; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            decode(ctx, toks.data() + n_prefix + i, 1, n_prefix + i, true);
            llama_synchronize(ctx);
            if (i >= 8) ms1 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            memcpy(ref.data() + (size_t) i*n_vocab, llama_get_logits_ith(ctx, 0), n_vocab*sizeof(float));
        }
        if (n_total > 8) printf("single token: %.2f ms per step\n", ms1/(n_total - 8));
        llama_free(ctx);
    }
    // B: windows
    llama_context * ctx = make_ctx(model);
    decode(ctx, toks.data(), n_prefix, 0, false);
    int pos = 0, bad_rows = 0, rows = 0;
    double win_ms[9] = {0}; int win_n[9] = {0};
    double max_diff = 0.0;
    for (size_t wi = 0; wi < wins.size(); ++wi) {
        const win w = wins[wi];
        const auto t0 = std::chrono::steady_clock::now();
        decode(ctx, toks.data() + n_prefix + pos, w.n, n_prefix + pos, true);
        llama_synchronize(ctx);
        if (wi >= sizes.size()) { win_ms[w.n] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); win_n[w.n]++; }
        const int keep = w.n - w.rollback;
        for (int r = 0; r < w.n; ++r) {
            // rows past the kept ones are compared too: they are valid continuations in the text
            if (pos + r >= n_total) break;
            const float * a = ref.data() + (size_t) (pos + r)*n_vocab;
            const float * b = llama_get_logits_ith(ctx, r);
            ++rows;
            if (memcmp(a, b, n_vocab*sizeof(float)) != 0) {
                double d = 0.0; int nd = 0;
                for (int v = 0; v < n_vocab; ++v) { const double e = fabs((double) a[v] - b[v]); if (e > d) d = e; nd += a[v] != b[v]; }
                if (d > max_diff) max_diff = d;
                if (bad_rows < 12) printf("window %zu (n=%d, rollback %d) row %d (pos %d): %d logits differ, max |diff| %.3g\n", wi, w.n, w.rollback, r, n_prefix + pos + r, nd, d);
                ++bad_rows;
            }
        }
        if (w.rollback > 0) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prefix + pos + keep, -1)) { fprintf(stderr, "seq_rm failed\n"); return 1; }
        }
        pos += keep;
    }
    for (int n = 2; n <= 8; ++n) if (win_n[n]) printf("window of %d: %.2f ms per step (%d steps, all logits read back)\n", n, win_ms[n]/win_n[n], win_n[n]);
    printf("%d rows in %zu windows after a %d-token prefix: %d rows differ from single-token decode, max |logit diff| %.3g\n", rows, wins.size(), n_prefix, bad_rows, max_diff);
    llama_free(ctx);
    llama_model_free(model);
    return bad_rows != 0;
}
