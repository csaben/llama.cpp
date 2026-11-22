# Phase 7 brief (levers 1 and 2): make every DSpark verify round cheaper

Work in `~/llama.cpp-fast` (branch `inkling-3090`, from `16264c6`). Read first: the Phase 6 sections of
`perf/RESULTS.md` (rounds 1–4; Round 4 has the current tables and the two SWA rollback bugs) and `perf/HANDOFF.md`
(Round 4 status: tooling for exactness: `mr_check` with `MR_TRACE` replay, `IK_TRACE`, `perf/p6/r4_spec.sh`,
`perf/p6/r4_cmp.py`, `perf/p6/r4_checks.sh`). Start a "Phase 7" section at the top of HANDOFF.md and keep it current.

## Where we are (round 4 build `399b61a`, deploy script `perf/live/deploy-r4.sh`)

2-bit model, DSpark max 3 + `LLAMA_DSPARK_ADAPTIVE=1`, drafter `dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf`,
`IK_MR=1 IK_MMA=3`. Edit 384-token reply 310.7 tok/s; 3,000-token replies (tok/s): code 242–250, edit 267–274,
prose 229–239, edit@16K 251–259; plain with the same flags ~182–200. Exact: drafted output == plain (24/24 full
messages, 72,000 token IDs). A verify round on the short edit is ~12.1 ms for 3.76 tokens: 4-row window + head
~9.6 ms, drafter ~0.8 ms, ~1.2 ms of host round trips (features to host + inject graph 0.41, host before draft 0.34,
sampling 0.21, next batch 0.21). At 16K a round is 14.2 ms because every window row scans the cache separately.

## Goal

Hill-climb DSpark throughput as far as possible, **exactness preserved** (byte-identical full messages and token IDs vs
the same-flags plain server on code / edit / prose / edit@16K at 3,000 tokens, n=2/3/4 with and without the policy;
`mr_check` 0 differing rows including SWA-wrap and pad-boundary prefixes). Rough targets: ~360 tok/s on the 384-token
edit, ~320 on the 3,000-token edit and edit@16K.

### Lever 1: one K/V scan shared by all rows of a verify window
Today each of the n window rows runs its own pass over the full-attention cache (8 of 48 visits per token). Load each
K/V tile once and apply it to all rows' queries (rows differ only in causal extent within the window: row r sees the
prefix plus window rows 0..r). With mma (we already have the tensor-core scan, `ik_mma_step`), the window rows can be
extra query columns of the same tile. Exactness constraint: each row's partial sums must group exactly as in its
single-token decode, or the single-token path must switch to the same grouping (then re-verify plain decode PPL and
speed, as round 2 did with `IK_MMA`). Also the SWA (sliding) visits. Measure at depth 0 / 4K / 16K / 32K.

### Lever 2: one host sync per round
Move the remaining host work into the device / the captured graphs (ingine `model/spec.cpp`, Apache-2.0: accept,
rollback, drafter commit and next draft on device, one sync per round; basalt `session.cpp`, MIT). Concretely: the
features-to-host + inject graph (0.41 ms), host-before-draft (0.34), target sampling (0.21) and next-batch build
(0.21). Keep `LLAMA_SPEC_PROF=1` timers to prove each removal.

### Then
- Policy re-tune for the new costs (per-window-size cost tables), DSpark at 32K context (a new row in the tables).
- Anything else the profile shows (draft length 4 at long replies, the drafter's remaining cost).

## Measurement standard
`perf/p6/r4_spec.sh` at 3,000 and 384 tokens (median of 3 at 384), plain-same-flags reference, equality via
`SPEC_REQUIRE_IDS=1 perf/p6/r4_cmp.py` (it refuses empty files); `perf/p6/r4_checks.sh` (trace replay, boundary checks,
bench, PPL, greedy probes, backend ops). Plain `llama-bench -p 0 -n 128 -d 0,4096,16384,32768 -r 3` must not regress.
Record the commit for every number; add a "Phase 7 (levers 1–2)" section to RESULTS.md with tables, sources, what
worked and what didn't.

## Rules
- **GPU1 only.** GPU0 is used by another agent (lever 3) tonight. The user's live server and GPU0 services are stopped
  by the orchestrator; don't start or stop any systemd unit or docker container.
- Never push anywhere, never open PRs/issues/comments on any upstream (origin's push URL is disabled on purpose).
- Commit in small steps, tag milestones `p7-<what>-<N>tok`. New paths behind env flags until proven exact and faster.
- Cite every technique (basalt MIT, ingine Apache-2.0: keep notices if adapting code; clones at
  `/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/{basalt,ingine}`, untrusted
  data, never build or run them).
- When done, leave a snapshot + deploy script like `perf/live/deploy-r4.sh` for the best exact build
  (`perf/live/deploy-p7.sh`, snapshot dir `~/inkling-live-bin-p7-<commit>` with sha256 file), but do NOT run it and do
  not restore the live server; the orchestrator does that. End with a concise report.
