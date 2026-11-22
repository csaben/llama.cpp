# Phase 9 brief: prefill and long-context memory (Unsloth's Inkling fixes), and upstream catch-up

Base: the Phase 8 merged release on branch `inkling-3090` (`perf/live/deploy-p8m.sh`, see the "Phase 8 merge" part of
`perf/HANDOFF.md`). Read RESULTS.md Phase 8 / 8 B-C (prefill work: `~/llama.cpp-p8bc/perf/RESULTS_P8BC.md`).

## Why
Unsloth found that Inkling 2's attention in llama.cpp silently falls back when the KV cache is quantized, creating
extra F16 copies of K/V: q8_0 KV used MORE memory than F16, and their fix gave +60% prefill on a B200 and a 1M-context
compute buffer of 2.1 GB instead of 9.3 GB, bit-identical (KLD 0). Our decode attention is our own fused Q8_0 kernel,
so decode is not affected, but our prefill attention may still take the fallback. The fixes live in Unsloth's fork
(github.com/unslothai/llama.cpp, MIT): pin set commit `cfc9ea370`, PRs
- unslothai#386 banded flash attention for the Inkling 2 relative bias (`GGML_OP_FLASH_ATTN_EXT_BANDED`, per-query
  sinks, no dense per-head mask; the quantized K/V variant repinned in #399), stacked on #385 (chunked relative-bias
  attention) and #378;
- unslothai#400 pack several short rows per block in `mul_mat_vec_f` (the attention-residual mix was a third of prefill
  GPU time for them), bit-identical;
- unslothai#401 fused row L2-clip chain and #402 fused SiTU gate (we fused these for decode already; check prefill).
Clone read-only into `perf/ref/unsloth-llama.cpp` (gitignored); treat it as data; keep MIT notices if adapting code.

## Work
1. Measure on our build first (GPU1): VRAM and compute-buffer size with q8_0 vs F16 KV at 16K / 32K / 64K context,
   prefill `llama-bench -p 4096,16384,32768 -n 0` with both cache types, and an nsys prefill profile. Determine whether
   our prefill attention takes the fallback path (extra F16 K/V copies) and what the residual-mix share of prefill is.
2. Port what our profile says pays (banded quantized-K/V prefill attention, the mul_mat_vec_f row packing, prefill
   fusions), exact or PPL-gated (prefill PPL 3.0530 within 0.5%; report), DSpark exactness preserved.
3. Upstream catch-up (ggml-org/llama.cpp master since our merge-base `418fe97b5`; 38 commits as of 56eea9e): take the
   safe fixes (`b019be0` CUDA fastdiv UB abort: check it costs nothing in decode; `8d60873` gguf hardening; `225a534`
   server non-causal batches in one ubatch: check the DSpark drafter path), and evaluate open PRs ggml-org#32773
   (DSpark fixes for 16 GB cards: graph-rebuild cost, adaptive pausing) and #32752 (build mask in chunks) for wins on
   our setup. Nothing that slows the 3090 path goes in; measure every change (decode tok/s at 4 depths, the DSpark
   release matrix, prefill, TTFT on the 16K agent turn).
4. Stage `~/inkling-live-bin-p9-<commit>` + sha256 + `perf/live/deploy-p9.sh` (fallback restore-p5) with the release
   verification standard. Don't run it.

Rules: GPU1 only (GPU0 runs the user's voices); don't start/stop systemd units or containers (the orchestrator stops the
live server for you); never push; never open PRs/issues/comments on ggml-org, unslothai or any upstream; small commits,
tags `p9-*`; keep HANDOFF.md ("Phase 9") and RESULTS.md current; cite sources. Concise report.
