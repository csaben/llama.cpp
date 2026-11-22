# Phase 8 B/C handoff

Branch `inkling-3090-p8bc` on GPU 0, RTX 3090 at 01:00.0. The finished B evidence is tagged `p8b-final`. The C kernels are tagged `p8c-packed`, `p8c-experts` and `p8c-bias`. The standard-microbatch release crossover and final C validation are running. [Results](RESULTS_P8BC.md).

## Stream B

The guarded DSpark policy selects bounded prompt-lookup continuations, corroborates low-margin model predictions and uses widths through six on strong copy runs. At both output budgets all 360 drafted responses matched the complete plain messages and 347,520 token IDs. The isolated agent measured 294.5 versus 285.5 tok/s frozen at 384 and 319.7 versus 320.1 at 3,000. Long prose gained 2.2%. New code was about 0.5% slower. GPU clock variation affected the long-context crossover. [Source and trials](RESULTS_P8BC.md#draft-selection-trials), [release matrix](RESULTS_P8BC.md#complete-phase-7c-request-matrices-on-gpu-0), [paired comparison](RESULTS_P8BC.md#frozen-release-crossover), [verifier checks](RESULTS_P8BC.md#stream-b-verifier-checks).

## Stream C

At microbatch 512 the B baseline measured 2,892 / 2,675 / 2,443 prompt tok/s at 4K / 16K / 32K. Its 16K agent turn reached the first streamed token in 6.820 seconds. The C changes unpack Q2_K in integer MMA fragments, schedule full expert tiles directly, and reuse relative-bias inputs across four heads. The 4K trials reached about 3,300 tok/s. Existing dense and FlashAttention tile widths won their trials. [Baseline profiles](RESULTS_P8BC.md#stream-c-baseline), [kernel and microbatch trials](RESULTS_P8BC.md#stream-c-kernel-trials).

## Reproduction

Run from the root of this worktree. Source `perf/p8bc/env.sh` before any GPU command. It overrides the legacy environment to select physical GPU 0 and sets the target, drafter and release flags. [Environment](p8bc/env.sh), [Phase 7c build flags](RESULTS.md).

- The complete B sweep is `bash perf/p8bc/matrix.sh build/p8-matrix`. Results include repeat-by-repeat identity checks. [B runner](p8bc/matrix.sh).
- For C use `PROFILE=1 PPL=1 perf/p8bc/prefill.sh final` and `perf/p8bc/ttft.sh final`. The latter streams uncached tool follow-ups on localhost port 18432. Set `DRAFT_WIDTH=6` to include the guarded drafter. Set `UB` for a different microbatch. The default is 512. [Benchmark](p8bc/prefill.sh), [server and timing](p8bc/ttft.sh).
- The saved B control libraries for this build are in `build/p8-b-final/`. To reproduce them, build `p8b-final` with the same flags and save `libggml-cuda.so.1`, `libllama.so.1` and `libllama-common.so.1` from that build. A C benchmark label beginning with `control` selects those libraries. Raw profiles, benchmark JSON, timings and server logs are in `build/p8-prefill/`. Compact evidence is in [perf/p8bc/prefill](p8bc/prefill/).

The first combined C run reproduced prefill PPL 3.0530 and reached 3,302 / 3,017 / 2,720 prompt tok/s. The 4K trace saved 181 ms of GPU kernel work at the same launch count. [Final profiles](RESULTS_P8BC.md#stream-c-quality-and-profiles).

The B/C/C/B crossover measured +13.5% / +12.4% / +11.6% throughput at 4K / 16K / 32K. Client TTFT fell from 6.827 to 6.008 seconds at 16K. All tests used microbatch 512. CUDA operation, bitwise-logit and integrated output validation follow this crossover. [Paired results](RESULTS_P8BC.md#stream-c-release-crossover).

Final C served outputs matched both C plain and the saved B plain captures on all eleven prompts at both budgets, 37,224 drafted IDs total. The CUDA, logit and rollback checks passed. Testing larger optional microbatches and end-to-end TTFT with DSpark. [Output hashes](p8bc/prefill/served.tsv), [bitwise logits](p8bc/prefill/logits.tsv).

The optional 2,048-token microbatch reached 5.553 s at 16K on the plain server, using 20,417 MiB and decoding the reply at 174.6 tok/s. The 512-microbatch run used 14,711 MiB and decoded at 181.0 tok/s. [Optional settings](RESULTS_P8BC.md#optional-microbatch-sizes).
