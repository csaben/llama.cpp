# Phase 6 plain decode handoff

Local branch inkling-3090-plain, initial tree f3c8d00. The calibrated Q2 file is present. GPU0 is the RTX 3090 at 01:00.0. Source perf/p5env.sh and then export CUDA_VISIBLE_DEVICES=0 GPU=0 before every GPU process. Build only in this checkout.

- Started the local sm_86 Release build with CUDA 12.8 nvcc and LLAMA_CURL=OFF, using the flags recorded in perf/RESULTS.md. Build log is build/plain-build.log.
- Next: record a GPU0 baseline and node-level Nsight Systems trace, compare expert kernels and short attention splits, then test overlap/fusion against the measured costs. Final checks include both PPL paths, affected backend ops and four decode depths.
