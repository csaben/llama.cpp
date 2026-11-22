# Phase 7b brief: consolidate (merge lever 3, fix the policy, stage one release)

Work in `~/llama.cpp-fast` (branch `inkling-3090`, HEAD `6ea9f5d`, staged release `59c71a9` / tag
`p7-release-edit-324tok`, deploy script `perf/live/deploy-p7.sh`). Read the Phase 7 sections of `perf/RESULTS.md` and
`perf/HANDOFF.md` first.

1. **Merge lever 3 round 2**: tag `p7l3r2-final-223tok` (`e03d993`) on branch `inkling-3090-lever3` (worktree
   `~/llama.cpp-lever3`; a Codex agent is still working there on GPU0: do not edit that worktree, only merge the tag).
   Its exact plain-decode changes: the post-attention 4-tap conv folded into the output projection, and the gate
   projection overlapped with attention (215.84 -> 223.07 tok/s at depth 0 on GPU0; 0 differing rows; DSpark rows
   exact on its branch). Resolve conflicts with phase 7's changes, rebuild, and re-prove exactness of the combined
   build with the phase 7 standard (`perf/p6/r4_cmp.py` with `SPEC_REQUIRE_IDS=1`, 3,000-token and 384-token runs,
   rollback/boundary/depth checks), plain `llama-bench -d 0,4096,16384,32768`, and PPL.
2. **Fix the policy for short prose**: phase 7 short prose is 172 tok/s with drafting vs 210 plain, and
   `LLAMA_DSPARK_ADAPTIVE=1` no longer protects it (172.8). Make the policy switch drafting off quickly when the
   acceptance it observes in the current request doesn't pay at the measured window costs (per-request warm-up,
   per-bin acceptance with fast decay, minimum confidence margin), without losing the edit/code gains (n3 fixed:
   324 / 287 / 277). Target: no configuration slower than plain on any prompt at either length, edits ≥ 320 at 384.
3. **Stage one release**: snapshot `~/inkling-live-bin-p7b-<commit>` with sha256 file and `perf/live/deploy-p7b.sh`
   (same pattern as deploy-p7.sh, with automatic fallback to `restore-p5.sh`). Do NOT run it.

Rules: GPU1 only (GPU0 is busy with lever 3); don't start or stop systemd units or containers; never push; never open
PRs/issues/comments anywhere; small commits, tags `p7b-*`; keep HANDOFF.md current ("Phase 7b"); add a "Phase 7b"
section to RESULTS.md with tables. Stop and report by 08:45 local at the latest.
