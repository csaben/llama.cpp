# Phase 8 brief (before the two-GPU reboot): squeeze the last of one GPU

Base: `~/llama.cpp-fast`, branch `inkling-3090`, release `p7c-release` (source `401460d`, live snapshot
`~/inkling-live-bin-p7c-401460d`, `perf/live/deploy-p7c.sh`). Read the Phase 7 / 7b / 7c sections of `perf/RESULTS.md`
and `perf/HANDOFF.md`, and lever 3's `~/llama.cpp-lever3/perf/RESULTS_LEVER3.md` (rounds 1–4, especially round 4's
ranked latency table `perf/p7l3r4/latency-single.md` there).

## Stream A: glue kernels' own latency (plain decode 239 -> as high as possible)
Plain decode is 238.7 tok/s (4.19 ms). Weight streaming is near its floor (one persistent kernel streams the 193
matvecs in 2.69 ms; the graph takes ~2.9). Graph edges cost 0.67 µs. The remaining ~0.95 ms to 300 tok/s is the glue
and attention kernels' own run time. Continue round 4's method: re-measure the ranked per-kernel table on this release
(depth 0 and 16K), then make the top kernels fast with exact changes (or PPL-gated, reported), one by one, paired ABBA
benchmarks. Every change must keep the DSpark verify path exact (window rows == single-token rows; release matrices).

## Rules
- **GPU1 only** unless told otherwise (GPU0 runs the user's voices). The live server is stopped by the orchestrator.
- Never push; never open PRs/issues/comments anywhere. Small commits, tags `p8a-*`. Keep `perf/HANDOFF.md` ("Phase 8")
  and a "Phase 8" section of RESULTS.md current; cite sources.
- When done, stage `~/inkling-live-bin-p8-<commit>` + sha256 + `perf/live/deploy-p8.sh` (fallback restore-p5), verified
  with the 7c release standard (3,000-token full-message + token-ID equality on all five prompts, 384 medians, rollback/
  boundary/depth checks, PPL, backend ops). Don't run it. Report concisely.
