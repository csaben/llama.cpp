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

static llama_context * make_ctx(llama_model * model, const char * rs_env = "MR_RS") {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = getenv("MR_CTX") ? atoi(getenv("MR_CTX")) : 8192; cp.n_batch = 2048; cp.n_ubatch = getenv("MR_UBATCH") ? atoi(getenv("MR_UBATCH")) : 512; cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq = getenv(rs_env) ? atoi(getenv(rs_env)) : 8; // MR_RS_A: of the single-token reference (a plain server has 0)
    cp.no_perf = true;
    cp.swa_full = getenv("MR_SWA_FULL") && atoi(getenv("MR_SWA_FULL")); // as llama-server: the SWA layers keep a ring, not the whole sequence
    cp.kv_unified = getenv("MR_KV_UNIFIED") ? atoi(getenv("MR_KV_UNIFIED")) != 0 : cp.kv_unified;
    return llama_init_from_model(model, cp);
}

static void decode(llama_context * ctx, const llama_token * toks, int n, int pos0, bool all_logits) {
    if (n == 0) return;
    if (n > 2048) { // a long prefix goes in batches, as a server does
        for (int i = 0; i < n; i += 2048) decode(ctx, toks + i, n - i < 2048 ? n - i : 2048, pos0 + i, all_logits);
        return;
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = all_logits || i == n - 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    llama_batch_free(b);
}

// MR_TRACE=<file written by a llama-server with IK_TRACE=file>: replay the decode / seq_rm calls of the server's target
// context (context B) and compare the logits of every kept output row with single-token decode of the final tokens
// (context A: the prompt in the server's batches, then one token per call).
struct tr_ev { char kind; std::vector<int> pos, tok, out; int p0 = 0, p1 = 0; };
static int replay(llama_model * model, const char * path) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ifstream in(path);
    std::string line, ctx_id, mem_id;
    std::vector<tr_ev> evs;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string k, a, b; ls >> k >> a;
        if (k == "D") {
            size_t n;
            if (!(ls >> b >> n) || n == 0) return 1;
            if (ctx_id.empty()) { ctx_id = a; mem_id = b; }
            if (a != ctx_id) continue;
            tr_ev e; e.kind = 'D';
            for (size_t i = 0; i < n; ++i) { int p, t, o; char c; if (!(ls >> p >> c) || c != ':' || !(ls >> t >> c) || c != ':' || !(ls >> o) || p < 0 || t < 0 || t >= n_vocab) return 1; e.pos.push_back(p); e.tok.push_back(t); e.out.push_back(o); }
            evs.push_back(e);
        } else if (k == "R" && a == mem_id) {
            tr_ev e; e.kind = 'R'; int seq, ok;
            if (!(ls >> seq >> e.p0 >> e.p1 >> ok)) return 1;
            if (ok) {
                if (seq != 0 || e.p1 != -1 || e.p0 < 0) return 1;
                evs.push_back(e);
            }
        } else if (k == "L" && a == ctx_id) {
            fprintf(stderr, "trace contains a state restore; its payload was not recorded\n");
            return 1;
        } else if (k == "C" && a == mem_id) {
            evs.clear(); // the run starts after the last clear
        }
    }
    // the prompt: the calls before the first call whose rows are all outputs
    size_t gen0 = 0;
    while (gen0 < evs.size() && !(evs[gen0].kind == 'D' && evs[gen0].out[0])) ++gen0;
    // the final tokens by position
    std::vector<int> fin;
    for (const auto & e : evs) {
        if (e.kind == 'R') { if (e.p0 >= 0 && (int) fin.size() > e.p0) fin.resize(e.p0); continue; }
        for (size_t i = 0; i < e.pos.size(); ++i) { if ((int) fin.size() <= e.pos[i]) fin.resize(e.pos[i] + 1); fin[e.pos[i]] = e.tok[i]; }
    }
    const int n_prompt = gen0 < evs.size() ? evs[gen0].pos[0] : 0;
    const int n_pos = (int) fin.size();
    const int lim = getenv("MR_TRACE_MAX") ? atoi(getenv("MR_TRACE_MAX")) : n_pos; // stop after this position
    if (gen0 == evs.size() || n_prompt >= std::min(lim, n_pos)) {
        fprintf(stderr, "no generation rows in trace range\n");
        return 1;
    }
    auto run = [&](llama_context * ctx, const tr_ev & e) {
        llama_batch b = llama_batch_init((int) e.pos.size(), 0, 1);
        for (size_t i = 0; i < e.pos.size(); ++i) { b.token[i] = e.tok[i]; b.pos[i] = e.pos[i]; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = e.out[i]; }
        b.n_tokens = (int) e.pos.size();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        llama_batch_free(b);
        llama_synchronize(ctx);
    };
    auto remove_suffix = [](llama_context * ctx, const tr_ev & e) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, e.p0, e.p1)) {
            fprintf(stderr, "replay could not rewind to %d\n", e.p0);
            exit(1);
        }
    };
    std::vector<float> ref((size_t) (std::min(lim, n_pos) - n_prompt)*n_vocab), got((size_t) (std::min(lim, n_pos) - n_prompt)*n_vocab);
    std::vector<int> step_of(n_pos, -1), row_of(n_pos, -1), size_of(n_pos, 0);
    {
        llama_context * ctx = make_ctx(model, getenv("MR_RS_A") ? "MR_RS_A" : "MR_RS");
        for (size_t i = 0; i < gen0; ++i) {
            if (evs[i].kind == 'D') run(ctx, evs[i]);
            else remove_suffix(ctx, evs[i]);
        }
        for (int p = n_prompt; p < n_pos && p < lim; ++p) {
            tr_ev e; e.kind = 'D'; e.pos = {p}; e.tok = {fin[p]}; e.out = {1};
            run(ctx, e);
            memcpy(ref.data() + (size_t) (p - n_prompt)*n_vocab, llama_get_logits_ith(ctx, 0), n_vocab*sizeof(float));
        }
        llama_free(ctx);
    }
    llama_context * ctx = make_ctx(model);
    for (size_t i = 0; i < evs.size(); ++i) {
        const tr_ev & e = evs[i];
        if (e.kind == 'R') { remove_suffix(ctx, e); continue; }
        if (e.pos[0] >= lim) break;
        run(ctx, e);
        if (i < gen0) continue;
        int o = 0;
        for (size_t r = 0; r < e.pos.size(); ++r) {
            if (!e.out[r]) continue;
            const int p = e.pos[r];
            const int output = o++;
            if (p < n_prompt || p >= n_pos || p >= lim) continue;
            memcpy(got.data() + (size_t) (p - n_prompt)*n_vocab, llama_get_logits_ith(ctx, output), n_vocab*sizeof(float));
            step_of[p] = (int) i; row_of[p] = (int) r; size_of[p] = (int) e.pos.size();
        }
    }
    // greedy consistency: the server's next token against the argmax of the single-token logits (a server at temperature 0)
    {
        int n_off = 0, first = -1; double first_gap = 0.0;
        for (int p = n_prompt; p + 1 < n_pos && p < lim; ++p) {
            const float * a = ref.data() + (size_t) (p - n_prompt)*n_vocab;
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (a[v] > a[best]) best = v;
            if (best != fin[p + 1]) { if (first < 0) { first = p; first_gap = a[best] - a[fin[p + 1]]; } ++n_off; }
        }
        printf("greedy check: %d of the server's tokens are not the argmax of the single-token logits", n_off);
        if (first >= 0) printf(" (first: the token at position %d, generated token %d, its logit is %.4g below the best)", first + 1, first + 1 - n_prompt, first_gap);
        printf("\n");
    }
    if (const char * dump = getenv("MR_DUMP")) { // position, then the top 3 (token, log-probability) of the single-token logits
        FILE * f = fopen(dump, "w");
        if (!f) { perror(dump); return 1; }
        for (int p = n_prompt; p < n_pos && p < lim; ++p) {
            const float * a = ref.data() + (size_t) (p - n_prompt)*n_vocab;
            float mx = a[0]; for (int v = 1; v < n_vocab; ++v) if (a[v] > mx) mx = a[v];
            double z = 0.0; for (int v = 0; v < n_vocab; ++v) z += exp((double) a[v] - mx);
            int top[3] = {-1, -1, -1};
            for (int k = 0; k < 3; ++k) { for (int v = 0; v < n_vocab; ++v) { if (v == top[0] || v == top[1]) continue; if (top[k] < 0 || a[v] > a[top[k]]) top[k] = v; } }
            fprintf(f, "%d", p);
            for (int k = 0; k < 3; ++k) fprintf(f, " %d %.6f", top[k], (double) a[top[k]] - mx - log(z));
            fprintf(f, "\n");
        }
        fclose(f);
    }
    int bad = 0, rows = 0, hist[9] = {0}; double max_diff = 0.0;
    for (int p = n_prompt; p < n_pos && p < lim; ++p) {
        if (step_of[p] < 0) continue;
        ++rows; hist[size_of[p] < 8 ? size_of[p] : 8]++;
        const float * a = ref.data() + (size_t) (p - n_prompt)*n_vocab;
        const float * b = got.data() + (size_t) (p - n_prompt)*n_vocab;
        if (memcmp(a, b, n_vocab*sizeof(float)) == 0) continue;
        double d = 0.0; int nd = 0;
        for (int v = 0; v < n_vocab; ++v) { const double x = fabs((double) a[v] - b[v]); if (x > d) d = x; nd += a[v] != b[v]; }
        if (d > max_diff) max_diff = d;
        if (bad < 8) {
            printf("pos %d (row %d of a %d-row step, trace event %d): %d logits differ, max |diff| %.3g; events before:", p, row_of[p], size_of[p], step_of[p], nd, d);
            for (int j = step_of[p] - 4 > 0 ? step_of[p] - 4 : 0; j <= step_of[p]; ++j) {
                if (evs[j].kind == 'R') printf(" rm>=%d", evs[j].p0); else printf(" D[%d..%d]", evs[j].pos[0], evs[j].pos.back());
            }
            printf("\n");
        }
        ++bad;
    }
    printf("replay of %s: prompt %d tokens, %d output rows up to position %d (rows by step size: 1:%d 2:%d 3:%d 4:%d 5+:%d): %d rows differ from single-token decode, max |logit diff| %.3g\n",
        path, n_prompt, rows, (lim < n_pos ? lim : n_pos) - 1, hist[1], hist[2], hist[3], hist[4], hist[5] + hist[6] + hist[7] + hist[8], bad, max_diff);
    llama_free(ctx);
    return bad != 0 || rows != std::min(lim, n_pos) - n_prompt;
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
    if (!model) return 1;
    if (getenv("MR_TRACE")) { const int r = replay(model, getenv("MR_TRACE")); llama_model_free(model); return r; }
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
    for (int n : sizes) if (n < 1 || n > 8) return 1;
    if (sizes.empty() || n_windows < 1 || n_prefix < 0) return 1;
    const bool rollbacks = !getenv("MR_ROLLBACK") || atoi(getenv("MR_ROLLBACK")) != 0;
    // MR_RAND=<seed>: a schedule like the server's policy: runs of single-token (plain) steps between windows of the
    // sizes in MR_SIZES (default 2,3,4), each window keeping 1..n rows (uniform); the rejected rows hold WRONG tokens
    // (as rejected drafts do) and are not compared. n_windows counts steps. MR_PLAIN_PCT (default 15): chance that a
    // step starts a plain run of 1..MR_PLAIN_RUN (default 24) tokens.
    const bool rnd = getenv("MR_RAND") != nullptr;
    if (rnd) {
        unsigned long long st = 0x9E3779B97F4A7C15ull ^ (unsigned long long) atoll(getenv("MR_RAND"));
        auto next = [&st]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (unsigned) (st >> 11); };
        if (!getenv("MR_SIZES")) sizes = {2, 3, 4};
        const int plain_pct = getenv("MR_PLAIN_PCT") ? atoi(getenv("MR_PLAIN_PCT")) : 15;
        const int plain_run = getenv("MR_PLAIN_RUN") ? atoi(getenv("MR_PLAIN_RUN")) : 24;
        if (plain_run < 1) return 1;
        while ((int) wins.size() < n_windows) {
            if ((int) (next() % 100) < plain_pct) {
                const int run = 1 + next() % plain_run;
                for (int j = 0; j < run && (int) wins.size() < n_windows; ++j) wins.push_back({1, 0});
            } else {
                const int n = sizes[next() % sizes.size()];
                wins.push_back({n, rollbacks ? (int) (next() % n) : 0});
            }
        }
    } else
    for (int i = 0; i < n_windows; ++i) {
        const int n = sizes[i % sizes.size()];
        wins.push_back({n, rollbacks && n > 1 && i % 3 == 1 ? 1 + (i % (n - 1 > 0 ? n - 1 : 1)) : 0});
    }
    int n_total = 0;
    for (auto & w : wins) n_total += w.n - w.rollback;
    if ((int) toks.size() < n_prefix + n_total + 16) { fprintf(stderr, "text too short\n"); return 1; }

    // A: single tokens
    std::vector<float> ref((size_t) n_total*n_vocab);
    {
        llama_context * ctx = make_ctx(model, getenv("MR_RS_A") ? "MR_RS_A" : "MR_RS");
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
        const int keep = w.n - w.rollback;
        llama_token wt[8];
        for (int r = 0; r < w.n; ++r) {
            wt[r] = toks[n_prefix + pos + r];
            if (rnd && r >= keep) wt[r] = (wt[r] + 977*(r + 1)) % n_vocab; // a rejected draft
        }
        decode(ctx, wt, w.n, n_prefix + pos, true);
        llama_synchronize(ctx);
        if (wi >= sizes.size()) { win_ms[w.n] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); win_n[w.n]++; }
        for (int r = 0; r < (rnd ? keep : w.n); ++r) {
            // rows past the kept ones are compared too: they are valid continuations in the text
            if (pos + r >= n_total) break;
            const float * a = ref.data() + (size_t) (pos + r)*n_vocab;
            const float * b = llama_get_logits_ith(ctx, r);
            ++rows;
            if (memcmp(a, b, n_vocab*sizeof(float)) != 0) {
                double d = 0.0; int nd = 0;
                for (int v = 0; v < n_vocab; ++v) { const double e = fabs((double) a[v] - b[v]); if (e > d) d = e; nd += a[v] != b[v]; }
                if (d > max_diff) max_diff = d;
                if (bad_rows < 12) printf("step %zu (n=%d, rollback %d; previous step n=%d, rollback %d) row %d (pos %d): %d logits differ, max |diff| %.3g\n", wi, w.n, w.rollback, wi ? wins[wi-1].n : 0, wi ? wins[wi-1].rollback : 0, r, n_prefix + pos + r, nd, d);
                ++bad_rows;
            }
        }
        if (w.rollback > 0) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prefix + pos + keep, -1)) { fprintf(stderr, "seq_rm failed\n"); return 1; }
        }
        pos += keep;
    }
    for (int n = 2; n <= 8; ++n) if (win_n[n] && !rnd) printf("window of %d: %.2f ms per step (%d steps, all logits read back)\n", n, win_ms[n]/win_n[n], win_n[n]);
    printf("%d rows in %zu steps (positions %d..%d) after a %d-token prefix: %d rows differ from single-token decode, max |logit diff| %.3g\n", rows, wins.size(), n_prefix, n_prefix + pos - 1, n_prefix, bad_rows, max_diff);
    llama_free(ctx);
    llama_model_free(model);
    return bad_rows != 0;
}
