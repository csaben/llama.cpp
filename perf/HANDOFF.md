# Phase 6 hand-off (keep current; newest status first)

Brief: `perf/PHASE6_BRIEF.md`. Environment: `. perf/p5env.sh` (GPU1 only, live flags, `$Q2`, `$DRAFT`). Raw outputs: `perf/p6/`.
Everything new is behind **`IK_MR=1`** (off by default; plain decode and the old batched path are unchanged without it).
The user's live server is stopped; at the very end run `bash perf/live/restore-p5.sh`.

## Design (implemented, WIP)

A "window" = a ubatch of 2..8 tokens of one sequence (a DSpark verify step). With `IK_MR=1` the graph
(`src/models/inkling2.cpp`, member `mr`, set in `init_fused()`) sends the window through the single-token fused ops, each
of which now takes the rows of the window; per row the arithmetic is the single-token kernel's, so logits are bit-identical
to decoding the tokens one at a time (checked with `perf/p6/mr_check.cpp`).

- **Dense matvecs** (`ggml/src/ggml-cuda/mmvq.cu`, `ik_mmvq_mr` / `ggml_cuda_ik_mmvq_mr`): up to 8 activation columns with
  the warps / blocks-per-thread / reduction order of `mul_mat_vec_q<type, 1>`; the weight block is read once for all columns.
  Requested per node with `op_params[15] == GGML_IK_MR_MAGIC` (set by `graph::mr_mark()`), dispatched at the top of
  `ggml_cuda_mul_mat` / `ggml_cuda_mul_mat_id` and allowed in `supports_op`.
- **Q|K|V projection of full-attention layers**: `ik_segmented_projection_rows` (`inkling-fmv.cu`), `op_params[14]` = rows
  of the Q group, `[13]` = n_tok; activation `[2304, n_tok, 3]`, output columns `q*n_tok + t`.
- **Grouped experts**: the same `ik_mmvq_mr` kernel with ids: grid y = (token, slot) pair; the first pair that selected an
  expert owns it and computes the columns of every pair of the window with that expert; other pairs exit. So each distinct
  expert of a visit is streamed once per window (gate|up Q2_K and down Q2_0).
- **Attention** (`inkling.cu`): `inkling_attn_kv_rows_kernel` (one block per kv head, a thread per K/V channel walks the
  rows: convolution over [state, rows so far], post-norm, Q8_0/F16 cache store, state rollback planes), then
  `inkling_attn2_kernel<..., MR=true>` with grid z = row (own K/V read back from the cache, no conv/store/state; per-row
  mask / rel_pos / kv_idx / part / counters offsets). A row whose mask shows nothing past `PAD(kv_idx+1, 256)` takes the
  splits of that smaller cache (so its partial sums group as in single-token decode) -- **this last bit is in flight**, see below.
- **Router**: `inkling_router2_kernel<NS, 0, MR=true>`, grid y = row. **moe_sum_conv**: `inkling_moe_sum_conv_rows_kernel`.
  **wo conv**: `shortconv_res_rows_f32` (a thread per channel walks the rows; used when `ids && n_t > 1`).
  **Controllers**: `ik_ctrl_apply8_rows`; no GEMM path and NS-specialized res_mix for windows when `IK_MR=1` (`ik_mr_enabled()`).
- **Recurrent (conv) state rollback**: row r of an n-row window writes plane `k = n-1-r` (cell `write_cell + k*plane_cells`)
  when `k < min(n, n_rs_seq+1)`, the layout of `build_conv_state_view`; attn op_params [14]/[15], moe_sum_conv [3]/[4].
- CPU references: window wrappers in `ggml/src/ggml-cpu/ops.cpp` (attn, router: row loop on shallow tensor copies;
  moe_sum_conv: row loop). ggml constructors take windows by shape (`ggml.c`: n_tok = mask->ne[1] / a->ne[2] / states->ne[1]).
- Debug: `IK_DBG_TAP=<tensor name>` (e.g. `l_out-9`) returns that tensor as the logits (graph ends there).

## Status

### Done / measured
- Baseline at d43d1d0: `perf/p6/base_decode.txt` (211.1 / 195.9 / 183.9 / 169.5 tok/s at depth 0/4K/16K/32K),
  `perf/p6/base_ppN.txt` (old batched step: pp2 8.45 ms, pp3 9.9, pp4 11.8, pp6 15.3, pp8 19.7 ms; plain token 4.74 ms).
- Profile of the old 4-token step (`perf/p6/prof.sh pp4 40 -p 4 -n 0 -r 20`): experts 4.9 ms (mul_mat_vec_q_moe, no
  dedup), dense ~2.3 ms, glue ~4.6 ms in 1,637 launches. A DSpark request (`perf/p6/prof_server.sh`): 19.5 ms wall per
  verify step; target head + drafter head matvecs (201K rows) 932 launches x 356 us = 3.2 ms per step: the drafter runs
  ~6 full-vocab correction matvecs + pool1d/argmax per step (`src/models/dspark.cpp` `build_inkling2_head` loop over the
  whole block, even with `--spec-draft-n-max 3`), then host sampling of the logits.
- `mr_check` (build line in the file header): **240 rows in 48 windows (sizes 2..8, with partial rollbacks) after a
  300-token prefix: 0 rows differ** from single-token decode; also exact at prefixes 250 / 1010 / 1270 / 2040.

### In flight
- Windows that cross a 256-cell pad boundary of the full-attention cache at n_kv >= 1536 (prefix 1525 / 1530 / 1786,
  `MR_SIZES=8 MR_ROLLBACK=0`) differed from single-token decode from layer 9 (first full-attention layer; found with
  `IK_DBG_TAP=l_out-9`). Fix being tested: per-row n_kv / splits in the MR attention kernel (the `cand` block at the top
  of `inkling_attn2_kernel`). If it still differs, compare `l_out-9` per row and look at how the tile grouping of
  `ik_mma_step` changes with `kv_per_split`.

### Next
1. Finish the boundary case above; run `mr_check` at prefixes 300 (48 windows), 1500 (36), 4000.
2. test-backend-ops cases for the window ops (INKLING_ATTN / INKLING_ROUTER / INKLING_MOE_SUM_CONV with n_tok > 1,
   MUL_MAT / MUL_MAT_ID with the magic) + existing cases still pass (`-o INKLING_ATTN`, `INKLING_ROUTER`,
   `INKLING_MOE_SUM_CONV`, `SHORTCONV_RES`, `LOOP_CTRL_APPLY`, `RES_MIX`, `MUL_MAT`, `MUL_MAT_ID`).
3. Measure: `IK_MR=1 build/bin/llama-bench -m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 2,3,4,6,8 -n 0 -r 20` against
   `perf/p6/base_ppN.txt`; `perf/p6/prof.sh` for the window profile; count distinct experts per window.
4. Server: `IK_MR=1 MODEL=$Q2 python3 perf/spec_probe.py dspark -- --backend-sampling --spec-type draft-dspark
   --spec-draft-model $DRAFT --spec-draft-n-max 3 --spec-draft-p-min 0`; compare the text with the plain run token for token.
5. Drafter (workstream 4): stop the Markov-head loop at n_max, fold `target.output x dspark_hidden_out` into one
   201K x rank matrix (basalt, MIT) or use a shortlist (ingine, Apache-2.0), one fused first-max argmax, return ids only.
6. Window policy (workstream 5), then RESULTS.md phase 6 section, plain-decode regression bench, PPL, greedy checks.
7. `bash perf/live/restore-p5.sh`.

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

- **Final tables and write-up done**: phase 5 section of `perf/RESULTS.md`, `INKLING.md` section, `NOTICE.inkling`,
  `perf/longctx.py`, `perf/spec_probe.py`, `perf/margin_probe.py`, `perf/kernsum_total.py`, `perf/live/switch-to-p5.sh`.
  Final 2-bit decode 210.32 / 194.88 / 182.46 / 168.67 tok/s at depth 0 / 4K / 16K / 32K; IQ4_NL dense 176.77 /
  166.09 / 157.28 / 146.88. Prefill 3213 / 3018 / 2782. Served 206.9 / 205.6 short; 16K prompt 6.27 s TTFT and
  179.4 tok/s; 32K prompt 13.47 s and 165.7 tok/s. PPL 3.0530 prefill / 4.2036 decode (2-bit).
- DSpark measured on the unchanged batched verify (slower than plain on code, edit, prose and a 16K edit; table in
  RESULTS). `--spec-draft-p-min` > 0 disables drafting entirely.
- Live server restored with `bash perf/live/restore-live.sh` (old snapshot build; switching is the user's call).

### In flight
- Nothing.

### Next (for whoever continues)
1. Grouped-expert, fused multi-token verify (workstream 2): not started. Per 4-token verify step today: expert
   matvecs 4.8 ms, dense matvecs incl. drafter 4.0 ms, unfused glue ~8 ms (nsys of a DSpark request). Target: a
   4-token window at <= ~9 ms including the drafter.
2. Scan: ~680 instructions per 16 cells x 1 head; ideas: compute P*scale once in the lane that owns P and shuffle it,
   32-bit row offsets, a cheaper mask/rel_pos path (CPU builds both per token, 0.12 ms at 32K).
3. Prefill: `mul_mat_q` is 57% of what is left, then flash attention 8.7% and `rel_bias` 6.5%.
4. `-ub 2048` decodes 4% slower after the prompt than `-ub 512` (unexplained) and takes 5.6 GB more VRAM.
