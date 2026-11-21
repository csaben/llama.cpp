# Phase 5 hand-off (keep current; newest status first)

Branch `inkling-3090`. Environment for every command: `. perf/p5env.sh` (GPU1, live flags, `$Q2`/`$IQ4`/`$Q6`/`$DRAFT`).
Raw outputs go to `perf/p5/`. Reference checkout (MIT): `git clone https://github.com/alesha-pro/maya-inkling-mini`
(study `docs/INKLING2-MINI.md`, `src/ik2/ik2_kernels.cu` `attn_decode_mma_kernel` at ~line 1383).

## Status

### Done
- Baseline at c3ab089 (2-bit model, Q8_0 KV), `llama-bench -p 0 -n 128 -d 0,4096,16384,32768 -r 3`:
  209.47 ± 0.30 / 168.69 ± 4.41 / 148.92 ± 3.74 / 131.17 ± 3.21 tok/s (perf/p5/base_decode.txt).
  Step time 4.77 / 5.93 / 6.71 / 7.62 ms. 16K->32K costs 0.91 ms for 285 MB of Q8 KV (8 full visits x 2176 B per
  cell) = 313 GB/s; the scan is compute-bound (per-cell butterfly shuffles + expf in every lane), not memory-bound.
  `IK_ATTN_CELLS_PER_SPLIT=128/256/512` does not help (<= +5% at 4K, nothing at 16K/32K).
- Model facts: n_head 24, n_head_kv 8 (G = 3), head dim 128, n_rel 16, extent 1024, SWA window 1024; full-attention
  layers are visited 8 times per token; SWA cache is separate (hybrid iSWA memory), KV rotation is off for this arch.

- **Tensor-core scan landed** (commit "inkling_attn: tensor-core (mma.sync) scan...", tag `p5-mma-attn-d32k-164tok`).
  `IK_ATTN_MMA=0` turns it off; `IK_ATTN_MMA_SPLITS` (10) / `IK_ATTN_MMA_SPW` (1) / `IK_ATTN_MMA_MIN` (1) tune it.
  2-bit model, Q8_0 KV, depth 0 / 1024 / 4096 / 16384 / 32768: 209.56 / 199.88 / 192.47 / 179.74 / 164.01 tok/s
  (before: 209.47 / 191.8 / 168.69 / 148.92 / 131.17). Sweep: splits 5/20/40 and spw 2/4 are all worse than 10/1.
  Kernel uses 181 registers => 1 block per SM, so 80 blocks is one wave.
  Checks: `test-backend-ops -o INKLING_ATTN` 16/16; decode PPL (2x1024, -ub 1) 4.2036 vs 4.2080 with the scan off;
  827-token mixed-prefill probe identical to scan-off and to the unfused reference; greedy gen2 identical, gen1/gen3
  diverge at near-ties (top-2 margins 0.068 and 0.040, `perf/margin_probe.py`).
  16K->32K now costs 0.53 ms for 285 MB = 535 GB/s (was 313).

### In flight
- Scan tuning: issue the next step's loads before computing the current one (registers are free up to 255), fewer
  redundant scale-word loads. Then opt-in lower-bit KV.

### Next
1. Opt-in lower-bit KV (Q4_0 / FP8) with its own PPL rows.
2. Workstream 2 (grouped-expert DSpark verify), workstream 3 (prefill).
3. Final tables in RESULTS.md (decode d0/4K/16K/32K, prefill 512/4096/16384, served short + long, PPL, spec), then
   `bash perf/live/restore-live.sh`.
