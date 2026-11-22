# Phase 7c brief: final release of the night (more DSpark, then merge lever 3 rounds 3–4)

Work in `~/llama.cpp-fast` (branch `inkling-3090`, release tag `p7b-release`, snapshot `~/inkling-live-bin-p7b-a9cdc61`,
`perf/live/deploy-p7b.sh`). Read the Phase 7 and 7b sections of `perf/RESULTS.md` and `perf/HANDOFF.md`.

Timeline (local time), hard: **by 09:00 a staged, verified release and a report**, so the orchestrator can deploy before
10:00. Until lever 3 round 4 finishes (its unit `inkling-codex-l3`; it reports by 07:30), work on part A; then do part B.

**Part A (GPU1, until lever 3 is done):** push DSpark throughput further on code and prose without losing edits:
draft acceptance (draft length per request from observed acceptance, confidence threshold sweeps), the drafter's
remaining cost, and the verify window cost at n=2/3. Keep everything exact. Commit only proven, exact wins.

**Part B (after `systemctl --user is-active inkling-codex-l3` is inactive):** merge branch `inkling-3090-lever3` up to its
final tag (`p7l3r4-*` if round 4 produced one, else `p7l3r3-final-225tok`); its changes are exact on its branch (controller
fusions removing 70 launches/token; round 4 glue-kernel latency fixes). Rebuild; re-prove the combined build with the 7b
standard: 3,000-token full-message + token-ID equality vs the same-flags plain server (all prompts incl. edit@32K,
n=2/3/4 ± policy), 384-token medians, rollback/boundary/depth checks, plain `llama-bench -d 0,4096,16384,32768`, PPL,
backend ops. Then stage `~/inkling-live-bin-p7c-<commit>` + sha256 + `perf/live/deploy-p7c.sh` (fallback to restore-p5)
WITHOUT running it. If the merge is not proven by 08:50, leave p7b as the release and say so.

Rules: GPU1 only until lever 3 is done (then GPU0 is also free, but the orchestrator will restart the user's services on
it at ~09:30, so finish GPU0 use by 09:15); don't start/stop systemd units or containers; never push; never open
PRs/issues/comments anywhere; small commits, tags `p7c-*`; keep HANDOFF.md ("Phase 7c") and RESULTS.md current.
