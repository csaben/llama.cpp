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

- **Scan diet + experiments** (commit "inkling_attn: scan as init/step/finish, branchless limbs..."):
  depth 0 / 1024 / 4096 / 16384 / 32768 = 210.90 / 200.67 / 194.72 / 182.10 / 168.58 tok/s (2-bit, Q8_0 KV, -r 3).
  What did NOT help (all measured, see RESULTS "what didn't work"): next-step load prefetch (250 regs, no gain);
  forcing 2 blocks/SM (spills, -1.5%); a stand-alone scan kernel with 4-warp blocks (150-165 at 32K); the same with
  a block per cell range covering all kv heads (162); a shared-memory tile fed by 16-byte loads (158). The stand-alone
  kernel is kept opt-in (`IK_ATTN_SCAN_MIN=2048`). Knock-out runs (IK_ATTN_SKIP bits, removed again): without the
  scan the 16K->32K step costs 0.05 ms (CPU mask/rel_pos build + upload), K/V loads alone 0.53 ms, loads replaced by
  16-byte sequential loads 0.39 ms, no loads 0.31 ms. The loop is ~680 executed instructions per 16 cells x 1 head.
  ncu is not usable here (ERR_NVGPUCTRPERM: perf counters need admin); nsys works (needs QdstrmImporter step).
- **Prefill: loop controllers as GEMMs for batches** (tag `p5-prefill-ctrl-gemm-3062tok`): nsys showed
  loop_ctrl_apply/down = 35% of a 4096-token prefill (one warp per output element). `IK_CTRL_GEMM_MIN` (8 tokens;
  0 = off). pp512 / pp4096 / pp16384: 2294 / 2184 / 2040 -> 3254 / 3062 / 2812 tok/s; with `-ub 2048`: 3247 / 3563 /
  3260. Prefill PPL 3.0530 (was 3.0516). LOOP_CTRL_DOWN 14/14, LOOP_CTRL_APPLY 23/23 (new batch cases F32 + F16).
  Remaining prefill profile (1325 ms / 4096 tok): mul_mat_q 57%, flash_attn 8.7%, rel_bias 6.5%.

### In flight
- Served long-context script (`perf/longctx.py`), DSpark acceptance per prompt type, quality tables, RESULTS.md.

### Next
1. Final tables in RESULTS.md (decode d0/4K/16K/32K for Q2 and IQ4_NL, prefill 512/4096/16384, served short + long,
   PPL prefill + decode, greedy probes, 827-token probe, spec acceptance), then `bash perf/live/restore-live.sh`.
2. Not started: grouped-expert DSpark verify (workstream 2). It needs multi-token versions of every fused decode op
   (attention with in-window keys + state rollback, router, grouped expert matvec); estimate in RESULTS.
3. Ideas left: CPU-side mask/rel_pos build is O(n_kv) per token (0.12 ms at 32K, 2%); fewer instructions per scan
   step (source-lane P*scale, 32-bit row offsets); Q4_0 KV as an opt-in variant (needs a nibble row reader).
