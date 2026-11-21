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

### In flight
- Workstream 1: tensor-core (mma.sync m16n8k16) scan inside `inkling_attn2_kernel` reading the existing Q8_0 cache
  (int8 -> FP16 exact in registers, per-block scales applied in FP32 on the K side and folded into P on the V side,
  Q and P passed as two FP16 limbs so the result equals FP32 math up to summation order). No cache-format change, so
  no prefill/decode basis risk. Target: ~700+ GB/s on the scan => ~170 tok/s at 32K, ~183 at 16K.

### Next
1. Finish + test (`build/bin/test-backend-ops -o INKLING_ATTN -b CUDA0` with `CUDA_VISIBLE_DEVICES=1`), tune splits.
2. Opt-in lower-bit KV (Q4_0 / FP8) with its own PPL rows.
3. Workstream 2 (grouped-expert DSpark verify), workstream 3 (prefill).
4. Final tables in RESULTS.md, `bash perf/live/restore-live.sh`.
