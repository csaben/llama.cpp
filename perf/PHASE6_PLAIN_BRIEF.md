# Phase 6 (plain-decode stream) brief: 2-bit Inkling 2 Mini at 300 tok/s on one RTX 3090

You work in the git worktree `~/llama.cpp-plain` (branch `inkling-3090-plain`, forked from `inkling-3090` at 929ef5f).
A separate agent is working on speculative (DSpark) window verify in `~/llama.cpp-fast` (branch `inkling-3090`) at the
same time. Do not edit `~/llama.cpp-fast`. Read `perf/RESULTS.md` (phases 1–5; phase 6 is being written by the other
agent), `INKLING.md`, and `perf/PHASE6_BRIEF.md` (the "Why" section has the measured profile). Keep your own notes in
`perf/HANDOFF_PLAIN.md` (create it; keep it current after every meaningful step) and your results in a "Phase 6:
plain decode" section of `perf/RESULTS_PLAIN.md`.

## Goal

Plain single-token decode of the calibrated 2-bit model (`$Q2`, live flags from `. perf/p5env.sh`) at **≥ 300 tok/s**
at depth 0 (`llama-bench -m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -r 3`), and as fast as possible at
4K/16K/32K. Today: 209–211 / 195 / 184 / 169 tok/s.

## Why it should be possible

The model reads **2.42 GB per token** (dense IQ4_NL 1,135 MB, experts Q2_K/Q2_0 1,019 MB at 8/128, head 261 MB), a
2.7 ms floor at the measured 891 GB/s (~369 tok/s). 300 tok/s = 3.33 ms per token. Measured (nsys, 4.97 ms/token under
the profiler):

| Per token | ms |
|---|---:|
| Dense IQ4_NL matvecs + head (`ik_segmented_projection`) | 1.53 (~910 GB/s) |
| Expert matvecs (Q2_K + Q2_0 with ids) | 1.31 (~780 GB/s) |
| `inkling_attn2_kernel`, 48 calls at ~11 µs (short context: latency, not bytes) | 0.54 |
| Glue: router2, loop_ctrl_down, ctrl_apply8, res_mix, norms, shortconv, situ_q8_1, moe_sum | 1.29 |
| GPU idle | ~0.30 |

So ~2.1 ms per token is latency. Weight streaming must keep going while the dependent glue runs. Ideas (measure,
rank, then build):

- **Persistent / megakernel decode** for the layer body (or the whole token): one resident grid that streams the next
  matvec's weights (cp.async into shared memory or `prefetch.global.L2`) while a few blocks run the dependent glue,
  with grid-wide barriers (cooperative groups) between phases. Ampere has no PDL, so this is how to overlap. Prior art:
  Hazy Research "Look Ma, No Bubbles!" (https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) and the
  existing phase-2 "megakernel" notes in RESULTS.md.
- **Fold glue into the matvec kernels' prologue/epilogue** (norm/mix/quantize of the activation inside the consumer
  matvec; router top-k and moe_sum inside the expert kernels), cutting launches and round trips.
- **Short-context attention:** 11 µs per call is latency; a smaller grid or folding it into the QKV epilogue.
- **Expert matvecs** from ~780 toward ~900 GB/s (bytes in flight, tail effects with 8 experts × small rows).
- **L2 evict-first hints** on streamed weights (`createpolicy` + `L2::cache_hint`, sm_80+) so the small hot tensors
  stay resident; **launching step N+1 before the host reads token N** in llama-server.

Exact math first. Anything that changes numerics is allowed only if prefill PPL (8×2048, `perf/ppl.txt`) and
decode-path PPL (2×1024, `-ub 1`) stay within 0.5% of 3.0530 / 4.2036 and is reported as such.

## Rules

- **GPU0 only, for you:** `export CUDA_VISIBLE_DEVICES=0` AFTER sourcing `perf/p5env.sh` (which sets GPU1). The user
  lent GPU0 for this work: its voice and speech-to-text services are paused by the orchestrator. Two small idle
  processes stay on it (`twice-serve`, `agentpc-server`, ~1.4 GB); leave them alone. **Never use GPU1**: the DSpark agent
  benchmarks there. Measure your own baseline on GPU0 first (the two cards may differ by a few percent) and compare
  against that, not against GPU1 numbers.
- **Build in this worktree only** (`build/` here): `cmake -B build -DGGML_CUDA=ON
  -DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86` (+ whatever `~/llama.cpp-fast`'s
  build uses; check its `build/CMakeCache.txt`).
- **Profiling:** `~/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys` (the system /usr/bin/nsys can't import traces);
  ncu at `/usr/bin/ncu`.
- Don't start or stop any systemd unit or docker container. The user's live server is stopped by the orchestrator.
- New paths behind env flags until proven; default changes only if exact (or within the PPL gate) and faster.
- **Commits:** small steps on `inkling-3090-plain`, tags `p6p-<what>-<N>tok`. Never push anywhere; never open PRs,
  issues or comments on any upstream.
- Cite every technique's source in RESULTS_PLAIN.md.

## Finishing

Write the results section (table at depths 0/4K/16K/32K before vs after, PPL, what worked, what didn't, sources),
make sure the tree builds and `test-backend-ops` passes for the ops you touched, then stop. Don't restore or switch the
live server; the orchestrator does that.
