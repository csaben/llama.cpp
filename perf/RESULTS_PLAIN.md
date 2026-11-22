# Phase 6: plain decode

One RTX 3090, PCI 01:00.0, calibrated Q2 GGUF, three core passes and eight of 128 experts. All measurements used the live phase-5 flags, GPU0, Q8_0 K/V, flash attention and full GPU offload. The headline metric is llama-bench tg128 with three repetitions. Nsight Systems numbers below describe a separate instrumented run.

## GPU0 baseline

| Depth | 0 | 4096 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| Baseline, tok/s (mean +/- SD) | 206.77 +/- 2.66 | 192.09 +/- 4.26 | 177.93 +/- 2.95 | 165.01 +/- 1.72 |

8x2048 prefill PPL was 3.0530. With 2x1024 and ubatch 1, decode PPL was 4.2036. [Benchmark](p6p/baseline.txt)

A 1 GiB read measured 886-889 GB/s on this card. A GGUF inventory gives 726.76 MB of IQ4_NL attention matrices over 48 visits, 1019.22 MB of active expert weights, 260.53 MB of head weights, 56.62 MB of F32 router weights and 113.25 MB of F16 controllers. Controllers run on the two returning core passes. The 1135 MB dense entry in the brief overstates the IQ4_NL matvec traffic. The listed weights sum to 2.176 GB; other small weights, activations, cache and repeated transactions are additional. A sustained transfer rate alone cannot predict serial decode latency. [Read probe](p6p/read-bandwidth.cu) and [measurements](p6p/bandwidth.txt)

The baseline trace covers 119 consecutive head-to-head intervals after warm-up. Q2_K gate/up costs 0.854 ms/token, Q2_0 down 0.530, IQ4 expert matvecs 0.184, dense attention projections and head 1.366, attention 0.586, router 0.380 and residual mixes 0.283. The instrumented intervals averaged 5.173 ms. [Full trace summary](p6p/baseline-profile.txt)

## Trials

- Two Q2_0 output rows per warp reduced a down projection from 12.02 to 10.45 us over 44 calls/token. The two lanes per Q2_0 block retain the small-K MMVQ partial sums and their writer lane. All 64 full-vocabulary logit rows matched the reference after a 200-token prefix of perf/ppl.txt. One and four rows per warp and the corresponding Q2_K up kernels were less effective. This adapts the row loop in this fork's mmvq.cu (ik_mmvq_mr_pairs) and the existing Q2_0 vector dot. [Trace](p6p/warp-fixed-profile.txt)
- Specializing the existing attention kernel to 24 query heads, 8 KV heads, 16 relative features and a four-tap convolution reduced attention from 12.16 to 10.84 us/call. The complete logit rows changed, so the quality gate applies.
- Dedicated blocks fetched the next output matrix during attention using the backend's prefetch.global.L2 helper. Fetching 512, 2048 and 4096 KiB reached 207.47, 206.74 and 206.39 tok/s. Shape specialization alone reached 208.45. The existing persistent input-projection kernel, fused router/conv and fused controller normalization also lost their paired comparisons. [Screen](p6p/screen.txt), [phase 3](RESULTS.md), [Hazy Research on overlapping independent weight loads with computation](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) and [NVIDIA PTX prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)
- A two-limb INT8 query encoded each 32-feature group relative to its maximum and fed the packed Q8_0 keys straight into m16n8k32 integer MMA. It passed decode PPL at 4.2094 but lost at all four depths to the shape-specialized FP16 scan. Both use the phase-5 Q8_0 cache layout and the same probability-times-value calculation. [Phase 5 and its Maya attribution](RESULTS.md) and [NVIDIA MMA fragment layout](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
- One warp per expert inside a down-projection block allowed the weighted sum, four-tap convolution and residual to run in its epilogue. Tiles of 2, 4 and 8 channels reached 206.08, 203.62 and 204.94 tok/s between 206.01 and 204.77 controls. The first CUDA graph matcher never fired because the allocation plan reused a source; these numbers are from a second implementation using the model-level INKLING_MOE_DOWN op. Adapted from this fork's inkling_moe_down_kernel and vec_dot_q2_0_q8_1.
