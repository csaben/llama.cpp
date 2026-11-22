# Phase 8 B/C handoff

Branch `inkling-3090-p8bc` on GPU 0, RTX 3090 at 01:00.0. Both streams are complete locally. Use `p8c-final` for the combined source and evidence. `p8b-final` isolates Stream B. The C kernel changes are `p8c-packed`, `p8c-experts` and `p8c-bias`. [Results](RESULTS_P8BC.md).

## Stream B

The guarded DSpark policy selects bounded prompt-lookup continuations, corroborates low-margin model predictions and uses widths through six on strong copy runs. At both output budgets all 360 drafted responses matched the complete plain messages and 347,520 token IDs. The isolated agent measured 294.5 versus 285.5 tok/s frozen at 384 and 319.7 versus 320.1 at 3,000. Long prose gained 2.2%. New code was about 0.5% slower. GPU clock variation affected the long-context crossover. [Source and trials](RESULTS_P8BC.md#draft-selection-trials), [release matrix](RESULTS_P8BC.md#complete-phase-7c-request-matrices-on-gpu-0), [paired comparison](RESULTS_P8BC.md#frozen-release-crossover), [verifier checks](RESULTS_P8BC.md#stream-b-verifier-checks).

## Stream C

The Q2_K MMA fragment, routed tile scheduling and four-head relative-bias changes improved 4K / 16K / 32K prefill by 13.5% / 12.4% / 11.6% in a B/C/C/B crossover at microbatch 512. At 16K, uncached client TTFT fell from 6.827 to 6.008 s plain and from 6.983 to 6.131 s with guarded DSpark. Prefill PPL was 3.0530. Existing dense and FlashAttention tile widths won their trials. [Crossover](RESULTS_P8BC.md#stream-c-release-crossover), [profiles and quality](RESULTS_P8BC.md#stream-c-quality-and-profiles), [trials](RESULTS_P8BC.md#stream-c-kernel-trials).

On the combined build, all 22 served comparisons and 37,224 drafted IDs matched C plain and B plain. All 216 captured full-vocabulary FP32 rows matched B byte for byte. All 728 mixed-width target rows matched scalar decode. The 3,792 CUDA operation tests passed and focused memory and synchronization checks reported zero errors. [Served captures](p8bc/prefill/served.tsv), [logit hashes](p8bc/prefill/logits.tsv), [CUDA and rollback checks](p8bc/prefill/checks.txt), [sanitizer](p8bc/prefill/sanitizer.txt).

With a 64K slot, a 2,048-token target microbatch reduced 16K TTFT to 5.553 s plain. With DSpark at microbatch 512 it reached 5.814 s and used 22,343 MiB. Setting both microbatches to 2,048 exceeded GPU 0 memory. The split setting also matched same-flags plain for the agent and 32K edit at both output lengths. The 512 / 512 setting is the full release comparison. [Microbatch tradeoffs](RESULTS_P8BC.md#optional-microbatch-sizes), [integrated DSpark replay](RESULTS_P8BC.md#integrated-dspark-first-token-latency).

## Reproduction

Run from the root of this worktree. Source `perf/p8bc/env.sh` before any GPU command. It overrides the legacy environment to select physical GPU 0 and sets the target, drafter and release flags. [Environment](p8bc/env.sh), [Phase 7c build flags](RESULTS.md).

- The complete B sweep is `bash perf/p8bc/matrix.sh build/p8-matrix`. Results include repeat-by-repeat identity checks. [B runner](p8bc/matrix.sh).
- For C use `PROFILE=1 PPL=1 perf/p8bc/prefill.sh final` and `perf/p8bc/ttft.sh final`. The latter streams uncached tool follow-ups on localhost port 18432. Set `DRAFT_WIDTH=6` to include the guarded drafter. Set `UB` for a different microbatch. The default is 512. [Benchmark](p8bc/prefill.sh), [server and timing](p8bc/ttft.sh).
- The saved B control libraries for this build are in `build/p8-b-final/`. To reproduce them, build `p8b-final` with the same flags and save `libggml-cuda.so.1`, `libllama.so.1` and `libllama-common.so.1` from that build. A C benchmark label beginning with `control` selects those libraries. Raw profiles, benchmark JSON, timings and server logs are in `build/p8-prefill/`. Compact evidence is in [perf/p8bc/prefill](p8bc/prefill/).
