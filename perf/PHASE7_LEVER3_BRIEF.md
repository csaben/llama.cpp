# Phase 7 brief (lever 3): plain decode toward 300 tok/s with a persistent / overlapped decode

Work in the worktree `~/llama.cpp-lever3` (branch `inkling-3090-lever3`, from `16264c6`, which already contains the
phase 6 plain-decode gains: router warp top-k, short 256-cell attention schedule, two-row Q2_0 down). Another agent
works on DSpark in `~/llama.cpp-fast` on GPU1 at the same time: never edit that tree, never use GPU1.

Read: `perf/RESULTS_PLAIN.md` and `perf/HANDOFF_PLAIN.md` (in `~/llama.cpp-plain`, the previous plain-decode round:
what was tried and the overlap diagnostics), `perf/PHASE6_PLAIN_BRIEF.md` there, and the Phase 6 sections of
`perf/RESULTS.md` here. Keep notes in `perf/HANDOFF_LEVER3.md` and results in `perf/RESULTS_LEVER3.md`.

## Facts

- 2-bit model (`$Q2` after `. perf/p5env.sh`, then `export CUDA_VISIBLE_DEVICES=0 GPU=0`): 2.42 GB/token (dense IQ4_NL
  matrices 727 MB + controllers/router ~170 MB, experts 1,019 MB, head 261 MB). Bandwidth floor at 889 GB/s on GPU0 =
  2.7 ms (~369 tok/s). Current plain decode ~221 tok/s (4.5 ms) at depth 0.
- Previous round's diagnostics (GPU0): replacing all 193 quantized matvecs with plain streamed reads and zero outputs
  still took 4.11 ms/token (243 tok/s) in its best configuration; deleting the reads too: 450 tok/s (2.2 ms). So the
  dependent glue + launch chain costs ~2.2 ms and is not overlapped with weight streaming. A cooperative grid that
  replaced the three Q2 expert ops in 44 of 48 visits took 34.99 µs/visit vs 33.22 for three launches.

## Goal

≥ 300 tok/s plain decode at depth 0 (3.33 ms/token), and gains at 4K/16K/32K, keeping prefill PPL 3.0530 and
decode-path PPL within 0.5% of 4.1979 (exact math preferred; report any change).

## Approaches to hill-climb (measure, rank, iterate; abandon what loses)

1. **Weight prefetch during glue:** while the dependent small kernels of visit v run, have otherwise idle SMs (or a
   concurrent stream / graph branch) pull visit v's *next* matvec weights into L2 (6 MB L2: chunk it; prefetch.global.L2
   or loads with evict-last on the chunk), so the matvec that follows starts warm. The previous round tried 0.5–4 MiB
   alongside attention only; try it systematically around every glue gap and measure gap-by-gap.
2. **A true persistent per-token kernel** (Hazy Research, "Look Ma, No Bubbles!"
   https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles; ThunderKittens megakernels): resident blocks with an
   instruction queue; producer warps stream the next weights (cp.async to shared memory) while consumer warps finish
   the dependent glue; fine-grained flags/counters instead of grid-wide barriers. Start with one layer-visit and
   prove it beats the launch chain before scaling.
3. **Fold glue into matvec prologues/epilogues** so fewer dependent steps remain (norm/mix/quantize of activations
   inside the consumer matvec; router top-k + moe_sum inside the expert kernels; controller apply inside the projection
   epilogue).
4. Anything the per-gap profile shows (graph topology, stream priorities, launch configuration, L2 persistence windows
   via cudaAccessPolicyWindow for the small hot tensors).

## Measurement
`llama-bench -m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768 -r 3` on GPU0 against your own
fresh GPU0 baseline at `16264c6`; PPL (8×2048 `perf/ppl.txt`, decode-path 2×1024 `-ub 1`); `test-backend-ops` for
touched ops; nsys `~/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys`. Important: the DSpark verify path (`IK_MR=1`)
must stay exact and working; if you change single-token kernels, run `perf/p6/r4_checks.sh` or at least `mr_check`.

## Rules
- **GPU0 only** (`CUDA_VISIBLE_DEVICES=0`); the user's GPU0 services are paused by the orchestrator for tonight.
  Don't start or stop any systemd unit or docker container.
- Build only in this worktree: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=86` (+ flags from `~/llama.cpp-fast/build/CMakeCache.txt`).
- Never push; never open PRs/issues/comments upstream. Commit in small steps, tags `p7l3-<what>-<N>tok`.
- New paths behind env flags until proven; cite every source in RESULTS_LEVER3.md.
- Finish with a results section and a concise report; don't touch the live server.
