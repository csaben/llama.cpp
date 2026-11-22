# Phase 8 merge brief: one release before the two-GPU reboot

`~/llama.cpp-fast`, branch `inkling-3090`: stream A is staged (`p8a-release`, source `66eadaf`, snapshot
`~/inkling-live-bin-p8-66eadaf`, `perf/live/deploy-p8.sh`). Streams B and C finished in the worktree `~/llama.cpp-p8bc`
(branch `inkling-3090-p8bc`, tag `p8c-final` = `7a2cd12`, B alone `p8b-final` = `a419ec9`); read its
`perf/HANDOFF_P8BC.md` and `perf/RESULTS_P8BC.md`. B: prompt-lookup + DSpark drafting (short agent replay 285.5 -> 294.5,
exact on 360 replies). C: prefill +11.6–13.5% at 4K–32K (16K agent turn TTFT 6.83 -> 6.01 s), PPL unchanged.

1. Merge `p8c-final` into `inkling-3090` (resolve conflicts with stream A's glue-kernel changes; keep both).
2. Rebuild and re-prove the combined build with the 7c/8a release standard: 3,000-token full-message + token-ID
   equality vs same-flags plain on all five prompts (n=2/3/4 ± policy), 384-token medians, the agent replay, the
   rollback/boundary/depth checks, full-logit rows, PPL (prefill 3.0530; decode 4.1979 / 4.2038), backend ops, prefill
   `llama-bench -p 4096,16384 -n 0`, plain `llama-bench -d 0,4096,16384,32768`.
3. Choose the release flags (whether prompt lookup is on by default; the 2,048/512 microbatch only if it fits the 64K
   slot on GPU1 with headroom) and stage `~/inkling-live-bin-p8m-<commit>` + sha256 + `perf/live/deploy-p8m.sh`
   (fallback restore-p5). Do NOT run it.

Rules: **GPU1 only** (GPU0 runs the user's voices again); the live server is stopped; don't start/stop systemd units or
containers; never push; never open PRs/issues/comments anywhere; small commits, tags `p8m-*`; keep HANDOFF.md ("Phase 8
merge") and RESULTS.md current. Concise report with the release table (plain vs DSpark per prompt, 384 and 3,000) and
prefill/TTFT numbers.
