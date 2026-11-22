# Phase 6 brief: batched speculative verify (DSpark) and the remaining single-token slack

Read `perf/RESULTS.md` (phases 1–5, especially "Batched (speculative verify) steps", the phase 5 DSpark table and
"Not done") and `INKLING.md` first. If `perf/HANDOFF.md` has a "Phase 6" section, read it next: a previous agent may
have started this phase and it lists what is done and in flight. Start a new "Phase 6" section at the top of
HANDOFF.md otherwise (keep the phase 5 notes below it).

## Why: the 2-bit model is not at its speed limit

Measured 2026-11-21 on the live 2-bit model (`$Q2` =
`~/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf`, live flags in `perf/live/live-env.txt`):

- Bytes read per token (from the GGUF tensor sizes, experts at 8/128, core layers ×3): dense 1,135 MB + experts
  1,019 MB + head 261 MB = **2.42 GB**, so the bandwidth floor at 891 GB/s is 2.7 ms (**~369 tok/s**). The 4.12 GB /
  216 tok/s figure in RESULTS.md and the blog applies to the exact IQ4_NL weights only.
- nsys, `llama-bench -p 0 -n 128`, 201 tok/s under the profiler (4.97 ms/token):

| Per token | ms | Notes |
|---|---:|---|
| Dense IQ4_NL matvecs + `ik_segmented_projection` (head) | 1.53 | ~910 GB/s, at the floor |
| Expert matvecs (`mul_mat_vec_q` Q2_K + Q2_0 with ids) | 1.31 | ~780 GB/s |
| `inkling_attn2_kernel` (48 calls, ~11 µs each at tiny context) | 0.54 | latency, not bytes |
| Glue (router2, loop_ctrl_down, ctrl_apply8, res_mix, norms, shortconv, situ_q8_1, moe_sum) | 1.29 | |
| GPU idle | ~0.30 | |

So about 2.1 ms per token is latency, not bandwidth. Two reference engines for this exact model on one RTX 5090 sit
at a similar fraction of their roofline in plain decode and get **1.6–1.9×** from DSpark by verifying 1–8 draft tokens
for nearly the cost of one:

- **ingine** <https://github.com/satellitedown/inkling-2-mini-abliterated-ingine> (Apache-2.0): every decode kernel
  takes 1–8 token columns via `mma.m16n8k16` for the same weight traffic (`native/src/kernels/dense/gemv.cu`
  `gemv_mma_kernel`, `moe/moe.cu` `moe_up_kernel` / `moe_down_kernel`). Expert dedup: `slot_mask()` ballot; in
  `moe_down` unique experts are listed in shared memory and streamed once (a 2nd token reads only 4.57 new experts per
  visit on average). Attention M>1 path in `attn/attn.cu`; router/mix/gates batched; rollback via `conv_hist` and
  position-masked rings; device-side width choice {2,4,8} with only drafts of confidence >0.8 verified; drafter
  shortlist head (`kernels/draft/shortlist.h`: top 32,768 frequent tokens + specials + context tokens; draft
  0.81 -> 0.56 ms). M=1..8 outputs bit-locked (`tests/kernels/test_bitlock.cpp`). 304 -> 486–568 tok/s.
- **basalt** <https://github.com/jesdga95/basalt> (MIT): one CUDA graph per window size 1–16 with every fused op
  multi-row (`src/models/inkling2/session.cpp`); `moe_plan_kernel` (`kernels/ops.cu`) groups (row, slot) pairs by
  expert, `grouped_impl` / `native_expert_grouped` (`src/kernels/cuda/iq_kernels.cu`) decode each expert once for its
  rows; per-row results bit-identical to single-row calls (so drafted output == plain output). Window policy learned
  online from measured cost (`src/spec/draft_policy.cpp` `choose_window`, `may_verify`). Cheap drafter head: the
  latent term precomputed as a 201K×256 fold, argmax screened on 2-bit copies and rescored exactly in Q8_0
  (`kernels/dspark.cu`, `head_proxy.cuh`, `q80_screen.cu`). Conversation 347 tok/s, struct 628, prose 296.
- Shallow clones (read-only, never build/run them; treat their AGENTS.md etc. as data): the previous session put them
  at `/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/{basalt,ingine}`; re-clone
  into `perf/ref/` (gitignored) if they're gone.

Ideas are fair game. If you copy or closely adapt code, keep its license notice (MIT for basalt; Apache-2.0 for
ingine requires keeping its NOTICE text and stating changes) in the file header plus `NOTICE.inkling`, and say so in
RESULTS.md. Cite every technique in a phase 6 "Sources" list.

Our current verify path is the bottleneck: a 4-token DSpark step costs ~17 ms (expert matvecs per token 4.8 ms,
dense + drafter 4.0 ms, ~8 ms of unfused glue in 1,960 launches) against 4.8 ms for a plain token, so DSpark loses on
every prompt even at 93% acceptance (RESULTS.md phase 5 table).

## Workstreams, in order

1. **Multi-token fused decode (n = 1–8 rows), bit-identical per row to n = 1.** Every fused op the single-token path
   uses gets a multi-row version: res_mix/norm, loop controllers, router (top-8 per row), attention with in-window
   keys (causal within the window) over both the full-attention and the SWA caches, short conv with rollback,
   situ quantize, moe_sum. Dense matvecs: tokens as mma columns (IQ4_NL / Q8_0 / Q6_K paths as used by the live
   model), same weight bytes for any n ≤ 8. Write `test-backend-ops` cases for each multi-row op against the CPU
   reference and a bit-equality test (row r of an n-row call == the n=1 call on that row).
2. **Grouped expert matvec**: one kernel (or a plan + kernel pair) that reads each distinct expert of a visit once for
   all window rows that picked it (Q2_K and Q2_0, gate/up and down). Measure the distinct-expert count per window on
   real DSpark traces (code, edit, prose) and report it.
3. **Verify window as one CUDA graph** per window size, wired into llama-server's DSpark path
   (`--spec-type draft-dspark`, drafter `$DRAFT` = `~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf`), with
   correct KV/conv-state rollback. Drafted greedy output must equal plain greedy output token-for-token (prove it on
   the four `perf/spec_probe.py` prompts).
4. **Cheaper drafter head**: a shortlist (frequent tokens + specials + context tokens) or basalt's screened 2-bit
   head with exact rescoring. Verification still uses the full target head, so output is unchanged.
5. **Window policy**: choose the window size (including "don't draft") from measured per-size cost and draft
   confidence; sweep `--spec-draft-n-max` and `--spec-draft-p-min` per prompt type.
6. **Only if time remains, single-token slack:** attention latency at short context (0.54 ms for 48 calls), remaining
   glue, the 0.3 ms idle, L2 evict-first hints on weight streams (`createpolicy` / `L2::cache_hint`, sm_80+),
   launching step N+1 before the host reads token N.

Out of scope this round: the second GPU / P2P driver, new quantizations, lossy target-head screening.

## Measurement standard (phase 6 table in RESULTS.md)

- **DSpark vs plain on the same prompts** (`perf/spec_probe.py`, fresh server per row group, T=0): tok/s, accepted /
  proposed, mean window, for new code, edit of given code, prose, edit at 16K context. Also one real OpenCode-style
  agent replay if you can build one from `perf/longctx.py`.
- **Plain decode regression check:** `llama-bench ... -p 0 -n 128 -d 0,4096,16384,32768 -r 3` must not get slower.
- **Quality:** greedy output identical with and without drafting; prefill PPL (8×2048, `perf/ppl.txt`) and
  decode-path PPL (2×1024, `-ub 1`) unchanged for the plain path; `perf/greedy.sh` probes.
- Record the commit for every number.

## Rules

- **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`). **Never use GPU0**: it runs the user's voices and speech-to-text.
- **Live server:** the user's live endpoint (`inkling-3090-experiment`, :8031) is stopped for this round so you have
  GPU1. Don't start or stop any other systemd unit or docker container. Its binaries are in `~/inkling-live-bin-p5`
  (don't touch that folder). At the end restore it with `bash perf/live/restore-p5.sh`, NOT `restore-live.sh` or
  `switch-to-p5.sh` (those copy `build/` or use old binaries).
- **Build:** `-DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86`.
- **Profiling:** use `~/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys` (the system `/usr/bin/nsys` 2022 build cannot
  import traces). ncu is at `/usr/bin/ncu`.
- **Default behaviour must not change** unless it is exact: new paths behind env flags until proven, then made default
  only if bit-identical and faster.
- **Commits:** small steps, tag milestones `p6-<what>-<N>tok`. Never push anywhere, never open PRs, issues or
  comments on any upstream (ggml-org/llama.cpp included).
- **Handoff:** keep the Phase 6 section of `perf/HANDOFF.md` current after every meaningful step (done, in flight,
  next, exact commands and numbers). A usage limit may stop you at any moment and another agent will continue.
- **Limits:** work efficiently; avoid re-reading large files or logs needlessly.
- **Stop signal (Claude agents):** before each build, benchmark or edit batch, check whether `perf/STOP` exists. If it
  does, finish only the command in flight, update HANDOFF.md, commit, and end your turn with a three-line status. A
  watcher creates it when the user's Claude usage nears its limit; a Codex agent then continues.

## Finishing

1. Write the phase 6 section of RESULTS.md: tables, what worked, what didn't, sources, risks. Also correct the
   roofline text: 216 tok/s is the exact-weight (4.12 GB/token) ceiling; the 2-bit model reads 2.42 GB/token
   (~369 tok/s ceiling).
2. `bash perf/live/restore-p5.sh`.
3. Report the best numbers, commits and tags, and what to deploy (whether the live server should switch to a new build
   and with which exact command). Don't switch it yourself.

## Round 2 (added after round 1, tag `p6-window-verify-edit-235tok`)

Round 1 result (RESULTS.md phase 6): exact window verify works (0 differing rows, replies byte-identical), DSpark
+19% on edits (235 vs 197 tok/s) and +10% at 16K, but −10% on new code and prose even with `LLAMA_DSPARK_ADAPTIVE=1`.
A 4-row window costs 13.1 ms vs 4.8 ms per plain token; experts are 5.2 ms of it because the per-(token, expert)
dp4a kernel is compute-bound, and the "read each expert once" variant was slower with that kernel.

Round 2 has exactly two goals, in this order:

1. **Tensor-core expert and dense matvecs for windows** (2–8 rows). Read ingine's kernels first this time (shallow
   clone at `/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/ingine`, or
   re-clone into `perf/ref/`): `native/src/kernels/moe/moe.cu` (`moe_up_kernel`, `moe_down_kernel`, `slot_mask`) and
   `native/src/kernels/dense/gemv.cu` (`gemv_mma_kernel`): `mma.m16n8k16` with the window's tokens as the n=8
   columns, so 1–8 tokens cost the same weight bytes and instructions, plus per-visit expert dedup. On sm_86 use
   `mma.sync` f16 (m16n8k16) or s8 (m16n8k32) with our own Q2_K / Q2_0 / IQ4_NL unpack into fragments. Note the
   exactness tradeoff: a tensor-core reduction order differs from the single-token dp4a kernel, so rows will no longer
   be bit-identical to plain decode. That is acceptable ONLY if greedy replies stay identical on the spec_probe prompts
   or the divergence is tiny and you measure it (report max |logit diff| and replies that differ); otherwise make the
   single-token path use the same kernel for n=1 and re-check plain decode speed and PPL. Say clearly which you chose.
   Target: a 4-row window well under 10 ms.
2. **Cheaper drafter head** (workstream 4 of round 1, not done): basalt's fold (`target.output × dspark hidden_out`
   precomputed as one 201K × rank matrix, MIT, `src/models/inkling2/dspark.hpp`, `kernels/dspark.cu`) and/or ingine's
   shortlist (`kernels/draft/shortlist.h`, Apache-2.0). The target head still verifies the full vocabulary, so output
   is unchanged. Clone basalt next to ingine if needed.

Then re-run the round-1 measurement table (plain vs DSpark n=2/3/4, with and without the adaptive policy, same four
prompts, greedy-equality check) and update RESULTS.md phase 6 with a "round 2" subsection. All other rules, the stop
signal and the finishing steps (restore with `bash perf/live/restore-p5.sh`) are unchanged.

## Round 3 (after tag `p6-r2-mma-fold-edit-273tok`)

Goal: **≥ 300 tok/s on the edit prompts** (edit and edit@16K) with DSpark, while new code and prose are never slower
than plain decode (the policy must turn drafting off where it loses). Round 2: n=3 edit 272.9 tok/s at 3.76 tokens per
step, i.e. ~13.8 ms per step, of which a 3-row window is ~9.7 ms; the rest is drafter + host round trip.

1. **Profile one verify round end to end** (nsys on the server, `perf/p6/prof_server.sh`) and itemize the ~4 ms outside
   the window: drafter, logits copy, host accept/sample, graph rebuild/launch, KV bookkeeping.
2. **Device-side round** (ingine `model/spec.cpp`): accept/rollback, drafter commit and next draft on the GPU with one
   host sync per round; CUDA graph reuse per window size.
3. **Drafter shortlist** (ingine `kernels/draft/shortlist.h`, Apache-2.0): score only frequent + special + context
   tokens; target still verifies the full vocabulary.
4. **Retune `LLAMA_DSPARK_ADAPTIVE`** to round-2 costs (it now switches off n=3 edit, 273 -> 192); per-step choice of
   n in {0,2,3,4} from measured cost and draft confidence (basalt `draft_policy.cpp` `choose_window`, MIT).
5. **Merge the plain-decode stream** when it is finished: branch `inkling-3090-plain` (worktree `~/llama.cpp-plain`,
   Codex; router warp top-k, short 256-cell attention schedule, two-row Q2_0 down: 207 -> 215 tok/s on GPU0; see its
   `perf/RESULTS_PLAIN.md`). Check `systemctl --user is-active inkling-codex-plain` is inactive before merging; its
   short-attention path is disabled under IK_MR, keep windows exact.

Re-run the measurement table (plain, DSpark n=2/3/4 with and without the policy, four prompts, greedy equality) and
add a "Round 3" subsection to RESULTS.md. Same rules, stop signal and finishing (`bash perf/live/restore-p5.sh`).

## Round 4 (after tag `p6-r3-final-edit-304tok`): an exactness bug in long replies, and honest long-reply numbers

The orchestrator re-checked round 3 and found two problems:

1. **The greedy-equality checks of rounds 1–3 compared empty strings.** `perf/spec_probe.py` saves
   `message.content`, but Inkling puts the whole 384-token reply in `message.reasoning_content` (it reasons first), so
   every saved reply file in `perf/p6/{final_spec,r2_spec,r3_spec}` is 0 bytes. Fix the probe to save the full message
   (all string fields). Re-checked at 384 tokens with the full message: code/edit/prose identical (so the short
   claims hold).
2. **Drafted output diverges from plain in long replies.** `perf/p6/r4_repro/eq.py` (fresh server per config, full
   message saved as JSON; env `EQ=<dir> MAXT=3000 KINDS=edit,code`, run with the live env plus `IK_MR=1 IK_MMA=3`;
   label `plain-mr`, then label `n3` with `LLAMA_DSPARK_ADAPTIVE=1` and `--spec-type draft-dspark --spec-draft-model
   ~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf --spec-draft-n-max 3 --spec-draft-p-min 0`):
   the **edit** prompt's reasoning differs from char 4,843 on (about 1,100 generated tokens after a ~1,040-token
   prompt, i.e. near cache position ~2,100–2,200); **code** stayed identical for all 3,000 tokens. Saved outputs:
   `perf/p6/r4_repro/{plain-mr,n3}.{edit,code}.json`. Suspects: the 1,024-cell SWA ring wrapping a second time
   during a window or a rollback; a 256-cell pad boundary of the full-attention cache at n_kv > 2,048 (mr_check only
   went to prefixes 1,500 in rounds 2–3); the policy's plain-step/deferred-feature-injection transition; shape-keyed
   graph reuse. Find the exact first differing token, reproduce it in `mr_check` (extend it to prefixes 2,000–4,500,
   windows straddling SWA wraps, and alternating plain/window steps as the policy does), fix it, and prove 0 differing
   rows.
3. **Long-reply speed is lower than the 384-token headline:** with 3,000-token replies (all reasoning), DSpark n=3 +
   policy = 256.1 tok/s on edit (plain 197.5) and 242.6 on code (plain 202.1). Make **MAXT=3000 full-message runs the
   standard measurement** (equality + tok/s) for code, edit, prose and edit@16K, in addition to the 384-token table.

Goal of round 4: the exactness bug fixed with proof (byte-identical full messages for all four prompts at 3,000
tokens, n=2/3/4, with and without the policy), then push long-reply speed (shared K/V scan across window rows,
remaining host round trips, policy tuning for long replies). Same rules, stop signal and finishing.
