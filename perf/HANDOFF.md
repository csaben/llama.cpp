# Phase 7b hand-off

Phase 7b is staged locally at `~/inkling-live-bin-p7b-a9cdc61`, source `a9cdc61` (`p7b-release-policy`), release tag `p7b-release`. Merge `1a45f6d` contains lever 3 round 2 (`e03d993`) and Phase 7's four-row verifier. The release uses the original folded 32K drafter, an adaptive cap of three and one 64K slot on physical GPU1. The manual deploy script was syntax-checked and left unexecuted. It falls back to Phase 5 on startup or health failure. [Deploy script](live/deploy-p7b.sh), [nine-file manifest](p7b/snapshot.sha256).

| Staged cap 3, decode tok/s | Plain 384 | Policy 384 | Plain 3,000 | Policy 3,000 |
|---|---:|---:|---:|---:|
| Code | 210.6 | 247.7 | 201.4 | 266.4 |
| Edit | 200.1 | 339.8 | 197.0 | 294.3 |
| Prose | 209.2 | 212.9 | 200.5 | 286.6 |
| Edit at 16K | 183.1 | 286.7 | 182.3 | 289.7 |
| Edit at 32K | 170.9 | 249.9 | 169.2 | 294.0 |

All six guarded configurations exceeded same-flags plain for all five prompts at both lengths. All 120 drafted messages and 124,560 IDs matched. Five additional prose prompts also beat plain at each tested cap and both lengths, with identical IDs. The smallest measured median gain was 1.8% in the main short matrix and 0.7% on the additional prose. Individual runs varied. The cap-3 lighthouse repeats were 195, 213 and 218 tok/s against a 209.2 plain median. Forced fixed-width controls and the earlier candidate's short-prose regression are recorded in the results. [Tables, diagnostics and reproduction](RESULTS.md#phase-7b), [short matrix](p7b/release/384/), [long matrix](p7b/release/3000/), [snapshot probes](p7b/release/snapshot-probes/).

Paired plain decode improved 3.1-3.8% over frozen Phase 7 across depths 0/4K/16K/32K. The final default and release-flag batteries each passed 2,714 rollback/boundary rows, six greedy probes and 1,108 CUDA operator cases. Another 1,024 depth rows and 88 full-vocabulary FP32 rows matched byte for byte. PPL reproduced 3.0530 prefill and 4.1979 / 4.2038 decode. All changes and tags are local. [Paired benches](p7b/combined/), [final default checks](p7b/release/checks-default/), [final release checks](p7b/release/checks/).

# Phase 7 hand-off

The staged build is `59c71a9`, local tag `p7-release-edit-324tok`, on branch `inkling-3090`. All measurements used physical GPU1 (RTX 3090). The release uses a fixed three-token draft, four-row MMA verification, a folded 32K shortlist, GPU-resident embedding tables and one 64K slot.

| Edit workload | Round 4 staged adaptive n3 | Phase 7 staged fixed n3 | Same-flags Phase 7 plain |
|---|---:|---:|---:|
| 384 tokens, median of three | 310.7 | 324.1 | 200.6 |
| 3,000 tokens | 266.6 | 287.4 | 197.5 |
| 3,000 tokens at 16K | 250.5 | 277.3 | 183.0 |
| 3,000 tokens at 32K | n/a | 277.4 | 170.1 |

- Fixed n3 gave 261.8 tok/s on long code and 239.3 on long prose. Short prose favored plain (209.8 vs 172.2). The target graphs consumed most of the round. Exactly one CUDA synchronization occurred in each of 1,800 measured steady fixed-n3 rounds. The serialized draft and target alone exceeded each requested whole-reply time budget at the observed acceptance. A five-row pack improved the isolated 32K scan but reduced served throughput.
- Packing four query rows into one pass at each head matched single-token reductions for full and sliding attention. At initial depths 0 / 4K / 16K / 32K, separate rows cost 9.28 / 11.11 / 11.48 / 12.44 ms. Packed rows cost 9.38 / 10.87 / 11.21 / 12.09 ms. Device feature rows and draft IDs, schedulers cached by width and pinned graph inputs let the target verify before the host reads the draft. The target sampler determines the result, then the host applies the verified rollback boundary.
- All 120 drafted replies and 124,560 generated token IDs in the required matrices matched same-flags plain. The trace, pad-boundary and SWA rollback battery had 0 differing logits in 2,714 rows. The separate depth sweep had 0 in another 1,024. Six greedy probes matched Round 4, PPL reproduced 3.0530 prefill and 4.1979 / 4.2038 decode, and 1,108 backend checks passed.
- The snapshot is `~/inkling-live-bin-p7-59c71a9`. All nine hashes and the library search paths verified. The manual [deploy script](live/deploy-p7.sh) was syntax-checked and left unexecuted. Phase 7 touched no live service or container and made no remote repository changes.
- [Full results, sources, commands and tradeoffs](RESULTS.md#phase-7-levers-1-2), [long matrix](p7/release/3000/), [short matrix](p7/release/384/), [GPU timelines](p7/release/profile/), [plain and correctness checks](p7/release/checks/), [snapshot manifest](p7/snapshot.sha256).

Plain decode passed at all four required depths in default and MR/MMA modes. The final change automatically queues CUDA input copies for single-row Q8 Inkling attention. [Paired benchmarks](p7/release/bench/), [default checks](p7/release/checks-default/).

# Phase 6 hand-off (keep current; newest status first)

## Round 4 status (2026-11-22, complete)

Phase 5 is restored. The active `inkling-3090-experiment` service is running `~/inkling-live-bin-p5/llama-server` on GPU1. Its health check on port 8031 returned OK. The Round 4 snapshot is staged for manual deployment.

- **Inference build `399b61a`**, local milestone `p6-r4-final-long-edit-274tok`. Both padded KV extent (`dca9008`) and SWA ring rewind (`06d9abb`) are fixed. The old edit diverged at token 1689 after the first bad logit row at position 2561. Original trace: 163/1699 bad rows became 0/1699. Two ring-wrap tests went from 70/73 and 71/73 to zero. A new adaptive edit trace has 0/2999 bad rows, maximum logit error 0, and zero greedy-choice differences.
- **Exactness:** 24/24 at 3000 tokens, 24/24 at 384 tokens with three repeats, full JSON and token IDs. Reference: backend sampling, `IK_MR=1 IK_MMA=3`, no drafter. Flag-free plain has different long answers. The separate character-214 edit difference between plain sampler modes starts in prefill, and both modes agree with CUDA fusion disabled. See `perf/p6/r4_checks/sampling.txt`.
- **Final tok/s**, code / edit / prose / edit@16K: 3000-token plain-mr 200.0 / 196.1 / 199.9 / 181.8, fixed n3 249.8 / 273.6 / 228.7 / 259.3, adaptive n3 242.2 / 266.6 / 238.8 / 250.5. At 384 tokens the adaptive n3 edit reached 310.7; prose was 205.1 vs 209.3 plain. Grouping verify blocks by shared KV tile added 6.1% to long edit@16K in the paired trial. All tables, acceptance counts and mean widths are in RESULTS.md under Round 4.
- **Other checks:** PPL 3.0530 prefill; decode 4.1979 (default), 4.2038 (window flags). Six CLI greedy outputs match Round 3. Default plain A/B was within 0.06% at all four depths. 1108/1108 backend tests passed. A 64K slot matched all four short probes. Tool-history replay: 777 identical tokens, 192.5 vs 225.9 tok/s.
- **Manual deployment:** `cd ~/llama.cpp-fast && bash perf/live/deploy-r4.sh` switches to the staged `~/inkling-live-bin-p6r4-399b61a` snapshot with n3 adaptive, a 64K slot, folded shortlist and backend sampling on GPU1. Restore with `bash perf/live/restore-p5.sh`. Snapshot checksums are in `perf/p6/r4_checks/snapshot.sha256`.

Reproduce the 3000-token table with `bash perf/p6/r4_spec.sh OUTPUT`, and the short table with `SPEC_MAXT=384 SPEC_REPS=3 bash perf/p6/r4_spec.sh OUTPUT`. Compare with `SPEC_KINDS=code,edit,prose,edit@16K SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py OUTPUT plain-mr n2 n3 n4 n2-adapt n3-adapt n4-adapt`. Final captures are in `perf/p6/r4_final_3000/` and `r4_final_384/`. Run `bash perf/p6/r4_checks.sh` for the original trace, wrap, mixed-width, PPL, greedy and backend checks. RESULTS.md gives the command for the fresh adaptive trace.

## Round 3 status (2026-11-21, finished; numbers at `1c46feb`)

Brief: "Round 3" at the end of `perf/PHASE6_BRIEF.md`. Full write-up: RESULTS.md "Round 3". Raw outputs: `perf/p6/r3_*`.
Flags for DSpark: `IK_MMA=3 IK_MR=1` (+ `LLAMA_DSPARK_ADAPTIVE=1` for the policy); drafter file
`~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf` (fold + shortlist head).

Final numbers (median of 3, `perf/p6/r3_spec.sh`): plain 215 / 204 / 214 / 187 (code / edit / prose / edit@16K); max 3:
223 / 304 / 167 / 253; max 3 + policy: 225 / 303 / 206 / 251. 28 of 28 drafted replies identical to plain; windows 0 of 257
rows differ. Plain `llama-bench` 221.35 / 202.63 / 190.40 / 176.04 (default), 213.12 / 199.21 / 188.00 / 173.80 (DSpark flags).
PPL 3.0530 prefill; decode 4.1979 (default) / 4.2038 (DSpark flags). Live server restored with `bash perf/live/restore-p5.sh`.

Done (commits after `59449ce`, tags `p6-r3-pick-short-edit-306tok`, `p6-r3-final-edit-304tok`):
- Round profile: `LLAMA_SPEC_PROF=1` (`perf/p6/r3_prof.sh <n_max> "<kinds>"`; env `DRAFT_FILE`, `PROF=0`, any `IK_*` / `LLAMA_*`),
  `LLAMA_GRAPH_BUILD_PROF=1`. Round (edit, max 3, nsys): 13.11 -> 11.58 ms; table in RESULTS.
- `ggml_dspark_pick` (`GGML_OP_DSPARK_PICK`; `ggml.c`, `ggml-cpu/ops.cpp`, `ggml-cuda/softcap.cu`): draft tokens + margins on the
  device, returned through `h_nextn` (`llama-context.cpp` copies `min(rows, ggml_nbytes)`). `LLAMA_DSPARK_PICK=0` = round 2 path.
- Shortlist head: `perf/p6/shortlist_rank.py URL OUT.bin` (needs a running llama-server for /tokenize), ranking committed as
  `perf/p6/shortlist_rank.bin`; `.venv/bin/python perf/p6/dspark_fold.py $Q2 $DRAFT OUT.gguf perf/p6/shortlist_rank.bin 32768`.
  `LLAMA_DSPARK_SHORT=0` = full vocabulary. Ids are stored as F32 (an I32 weight with `GGML_OP_NONE` is skipped by the loader).
- Policy rewrite in `common/speculative.cpp` (`adaptive_begin`, `adaptive_window`, `defer_features`, `flush_pending`);
  `LLAMA_DSPARK_ADAPTIVE_LOG=1` prints steps by kind, costs and chances when the server exits.
- CUDA graph key includes the graph shape (`IK_GRAPH_KEY_SHAPE=0` = old key).
- Merge of `inkling-3090-plain`; attention: specialized instance also for windows and under `IK_MR` (`IK_ATTN_FIXED_MR=0` = generic),
  short schedule off under `IK_MR`.

Next, if someone continues (RESULTS "Not done"): one scan over K/V for all rows of a window (the 16K case: 253 tok/s); a
graph (scheduler) cache per window size in `llama_context` (0.88 ms per change of size); a device-side accept + drafter commit;
context tokens in the shortlist; a `test-backend-ops` case for `GGML_OP_DSPARK_PICK`; the unexplained slow server starts
(two of about twenty were 10% slower for their whole life).

## Round 2 status (2026-11-21, finished)

Brief: "Round 2" at the end of `perf/PHASE6_BRIEF.md`. New flag **`IK_MMA`** (off by default; `IK_MMA=3` is the setting to use,
together with `IK_MR=1`). Folded drafter file: `~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-Q2K.gguf` (`$DRAFT_FOLD`).

Done (commits on `inkling-3090` after `929ef5f`):
- `ik_mma_moe` (`mmvq.cu`): expert matvecs on tensor cores, `mma.sync m16n8k32` s8, tokens as the 8 columns, expert dedup by
  ballot over the ids (layout after ingine `moe_up_kernel`, Apache-2.0, credited in the file header and `NOTICE.inkling`).
  `IK_MMA=1`: Q2_K and Q2_0 experts. Correct to 1e-6 against the dp4a kernels (`IK_MMA_CHECK=1` prints the difference).
- Exactness: a different fp32 order changes logits by up to 3.7 (the model amplifies 1e-7 through routing and Q8_1
  re-quantization), so the single-token path takes the same kernels (`ggml_cuda_ik_mma_moe_ok`, `ggml-cuda.cu` MUL_MAT_ID).
  Windows are again bit-identical to single-token decode (`mr_check`: 0 rows differ).
- `ik_q2k_moe` (`IK_MMA=3`, bit 2): Q2_K experts on CUDA cores with exact block sums (one fp32 multiply-add per Q8_1 block,
  activation sums from a prologue, weights unpacked once for all tokens of the expert). Faster than the mma kernel on this
  card at 1 token (19.2 vs 20.9 us; dp4a kernel 18.6) and equal at 4 (56 us for 20 experts). Q2_0 stays on tensor cores
  (10.7 us at 1 token vs 11.25 dp4a; 29 us at 4).
- Window dense matvecs (`ik_mmvq_mr`, `ik_segmented_projection_rows`): IQ4_NL weights looked up once for all columns
  (`ik_iq4_nl_decode` / `ik_iq4_nl_dot` in `vecdotq.cuh`), bit-identical. Head 628 -> 388 us, 4300 rows 30 -> 22 us.
- Window cost (`MR_SIZES=n MR_ROLLBACK=0 mr_check ... 300 16`, `IK_MMA=3 IK_MR=1`): 2 / 3 / 4 / 8 rows = 6.65 / 8.63 / 9.8 / 19.1 ms
  (round 1: 8.92 / 11.02 / 13.13 / 22.88); single token 4.76 ms; plain `llama-bench` tg128 210.5 (IK_MMA=3) vs 210.9.
- Drafter head fold: `perf/p6/dspark_fold.py TARGET DRAFTER OUT` writes `dspark_head_fold.{weight (Q8_0 [256, n_vocab]), bias}`;
  `src/models/dspark.cpp` uses it when present (`LLAMA_DSPARK_FOLD=0` turns it off). Idea from basalt (MIT), no code copied.

Not done / dropped (measured): a block that walks several experts (59.6-70 us vs 56), two rows per lane, 5 blocks per SM by
`__launch_bounds__`, the min term as one f16 mma (no gain); tensor-core dense matvecs (the single-token dense kernels are at
the DRAM floor and their fp32 order cannot be reproduced from 32-wide sums; a different order is not acceptable, see above).

Final numbers (`d771a28`, RESULTS.md "Round 2"; raw: `perf/p6/r2_*`): DSpark max 3 code 202 / edit 273 / prose 167 / 16K 220
tok/s (plain 208 / 196 / 207 / 179), max 2: 216 / 255 / 183 / 212; 28 of 28 replies identical to plain; windows 0 of 457 rows
differ; plain `IK_MMA=3` 208.65 / 193.31 / 182.15 / 168.47 (default 209.98 / 194.52 / 182.60 / 168.11); PPL 3.0530 / 4.1935.
Live server restored with `bash perf/live/restore-p5.sh`.

Next, if someone continues: retune `LLAMA_DSPARK_ADAPTIVE` (it now turns drafting off where it wins); the drafter's base
logits (382 us 4-column pass over the head) and host scan (in-graph argmax + margin, or a shortlist); the DRAM streaming
of the 16-row expert tiles (620 GB/s at 4 rows); attention and router rows (1.75 ms of a 4-row window).

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

## Status: finished (2026-11-21)

Results, tables, what did not work, sources, risks and the deploy command: phase 6 section of `perf/RESULTS.md`.
Final numbers at `2359b45` (`perf/p6/final_spec.txt`, `final_decode.txt`, `final_ppl.txt`, `mr_check.txt`, `backend_ops.txt`).

- Windows of 2-8 rows are bit-identical to single-token decode (`mr_check`: 0 of 700 rows differ).
- DSpark max 3 with `IK_MR=1`: code 158, edit 235, prose 139, edit@16K 196 tok/s (plain 208 / 197 / 207 / 179);
  with `LLAMA_DSPARK_ADAPTIVE=1`: 187 / 237 / 188 / 191. Replies identical to plain in all 20 comparisons.
- Plain decode 209.9 / 194.1 / 182.6 / 168.3 tok/s, PPL 3.0530 / 4.2036 (unchanged).
- Tools: `perf/p6/mr_check.cpp` (equality + window timing; `MR_SIZES`, `MR_ROLLBACK`, `MR_UBATCH`), `perf/p6/prof.sh`,
  `prof_server.sh`, `prof_check.sh` (nsys tables), `IK_DBG_TAP`, `IK_MR_STATS`, `IK_MR_GROUPED`, `LLAMA_DSPARK_ADAPTIVE=2`.

### Next (if someone continues)
1. Drafter head: fold `target.output x dspark_hidden_out` into a 201K x rank matrix or a shortlist (1.34 ms -> ~0.3 ms
   per step at max 3); skip the feature injection while the policy is not drafting (3% of plain speed).
2. Expert matvecs are 5.2 ms of a 4-row window and bound by dp4a per (token, expert) pair. A tensor-core kernel needs
   weights stored for it (unpacking 2-bit blocks on the fly costs as much as dp4a) and must reproduce the float order
   of `mul_mat_vec_q<type, 1>` from exact integer group sums to stay bit-identical.
3. Host time per DSpark step is about 2.3 ms (gaps after the drafter graph: logits copy and scan, feature copies).
4. Window cases for INKLING_ATTN and marked MUL_MAT / MUL_MAT_ID in `test-backend-ops`.

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
