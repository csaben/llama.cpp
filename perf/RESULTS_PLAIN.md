# Phase 6: plain decode

One RTX 3090, PCI 01:00.0, calibrated Q2 GGUF, three core passes and eight of 128 experts. All measurements used the live phase-5 flags, GPU0, Q8_0 K/V, flash attention and full GPU offload. The headline metric is llama-bench tg128 with three repetitions. Nsight Systems numbers below describe a separate instrumented run.

## GPU0 results

| Cache depth | Baseline tok/s | Final tok/s | Gain |
|---:|---:|---:|---:|
| 0 | 206.77 +/- 2.66 | **215.38 +/- 1.33** | +4.2% |
| 4096 | 192.09 +/- 4.26 | **197.72 +/- 3.40** | +2.9% |
| 16384 | 177.93 +/- 2.95 | **185.58 +/- 2.33** | +4.3% |
| 32768 | 165.01 +/- 1.72 | **171.31 +/- 1.95** | +3.8% |

Values are means and sample standard deviations. The original GPU0 baseline is the comparison used throughout. A second four-depth control disabled all three optimized operators in the final binary to check for clock and run-order effects. The 300 tok/s target requires 3.333 ms/token, versus 4.643 ms measured at depth 0. [Original baseline](p6p/baseline.txt), [final run](p6p/final.txt) and [same-binary control](p6p/paired-control.txt)

The two quality gates used the same corpus and commands on both revisions.

| Perplexity path | Baseline | Final | Relative change |
|---|---:|---:|---:|
| Prefill, 8 x 2048 | 3.0530 | 3.0530 | 0.00% |
| Single-token decode, 2 x 1024 with ubatch 1 | 4.2036 | 4.1979 | -0.14% |

The independent logit comparison used a 200-token prefix from the corpus and then 64 teacher-forced single-token steps. With the new expert kernel and router alone, all 64 full-vocabulary rows matched bit for bit. Adding the attention changes preserved 63 of 64 top tokens and 2 of 64 full rows. The maximum absolute logit change was 4.99024. CUDA0 backend tests passed 16/16 INKLING_ATTN, 11/11 INKLING_ROUTER and 1081/1081 MUL_MAT_ID. The complete sm_86 Release build succeeded. [Validation output](p6p/validation.txt)

The final node trace measured 10.08 us for expert down, 9.11 us for attention and 7.29 us for the router, compared with 12.05, 12.22 and 7.91 us on the initial trace. Their combined saving is about 0.266 ms/token. [Final trace](p6p/final-profile.txt) and [baseline trace](p6p/baseline-profile.txt)

A 1 GiB read measured 886-889 GB/s on this card. A GGUF inventory gives 726.76 MB of IQ4_NL attention matrices over 48 visits, 1019.22 MB of active expert weights, 260.53 MB of head weights, 56.62 MB of F32 router weights and 113.25 MB of F16 controllers. Controllers run on the two returning core passes. The 1135 MB dense entry in the brief overstates the IQ4_NL matvec traffic. Those weights sum to 2.176 GB. Small tensors, activations, cache and repeated transactions add traffic. A sustained transfer rate alone cannot predict serial decode latency. [Read probe](p6p/read-bandwidth.cu) and [measurements](p6p/bandwidth.txt)

The baseline trace covers 119 consecutive head-to-head intervals after warm-up. Q2_K gate/up costs 0.854 ms/token, Q2_0 down 0.530, IQ4 expert matvecs 0.184, dense attention projections and head 1.366, attention 0.586, router 0.380 and residual mixes 0.283. The instrumented intervals averaged 5.173 ms. [Full trace summary](p6p/baseline-profile.txt)

## Trials

- Two Q2_0 output rows per warp reduced a down projection from 12.02 to 10.45 us over 44 calls/token. The two lanes per Q2_0 block retain the small-K MMVQ partial sums and their writer lane. All 64 full-vocabulary logit rows matched the reference after a 200-token prefix of perf/ppl.txt. One and four rows per warp and the corresponding Q2_K up kernels were less effective. The final kernel interleaves both row accumulators before reducing either row. Separating them cost about 2 us/visit. [Early trace](p6p/warp-fixed-profile.txt), [row loop in ik_mmvq_mr_pairs](../ggml/src/ggml-cuda/mmvq.cu) and [Q2_0 dot](../ggml/src/ggml-cuda/vecdotq.cuh)
- Specializing the existing attention kernel to 24 query heads, 8 KV heads, 16 relative features and a four-tap convolution reduced attention from 12.16 to 10.84 us/call. The complete logit rows changed, so the quality gate applies.
- Dedicated blocks fetched the next output matrix during attention using the backend's prefetch.global.L2 helper. Fetching 512, 2048 and 4096 KiB reached 207.47, 206.74 and 206.39 tok/s. Shape specialization alone reached 208.45. The existing persistent input-projection kernel, fused router/conv and fused controller normalization also lost their paired comparisons. [Screen](p6p/screen.txt), [phase 3](RESULTS.md), [Hazy Research on overlapping independent weight loads with computation](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) and [NVIDIA PTX prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)
- A two-limb INT8 query encoded each 32-feature group relative to its maximum and fed the packed Q8_0 keys straight into m16n8k32 integer MMA. It passed decode PPL at 4.2094 but lost at all four depths to the shape-specialized FP16 scan. Both use the phase-5 Q8_0 cache layout and the same probability-times-value calculation. [Phase 5 and its Maya attribution](RESULTS.md) and [NVIDIA MMA fragment layout](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
- One warp per expert inside a down-projection block allowed the weighted sum, four-tap convolution and residual to run in its epilogue. Tiles of 2, 4 and 8 channels reached 206.08, 203.62 and 204.94 tok/s between 206.01 and 204.77 controls. The first CUDA graph matcher never fired because the allocation plan reused a source; these numbers are from a second implementation using the model-level INKLING_MOE_DOWN op. [Measured model-level runs](p6p/trials.txt), [inkling_moe_down_kernel](../ggml/src/ggml-cuda/inkling.cu) and [vec_dot_q2_0_q8_1](../ggml/src/ggml-cuda/vecdotq.cuh)
- With the minimum 256-cell cache, one block per KV head is enough when eight warps scan 16-cell tiles in round-robin order. A block-wide mask assembled with warp ballots skips tiles containing only padding or the separately evaluated current token. Five repetitions in off/on/on/off order gave 208.25 / 212.78 / 212.61 / 208.26 tok/s. A separate block for the current K/V update helped at depth 0 but lost 3-4 tok/s at larger depths. At 4K/32K, ten splits reached 195.44/169.81, five reached 194.44/166.46, twenty reached 190.07/165.80 and thirty-two reached 183.80/159.91. The shipped long scan keeps the existing split limit. [Measurements](p6p/trials.txt), [attention kernel and phase-5 scan](../ggml/src/ggml-cuda/inkling.cu) and [NVIDIA warp vote instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync)
- For the 128-expert router, each of four warps sorts 32 scores and contributes its top eight. A final warp merges the winners. The key breaks equal scores by expert ID, and normalization adds the selected probabilities in original rank order. The two paired trials gained 1.12 and 1.30 tok/s. A separate router kernel overlapping the single-state normalization and gate dot products yielded no further consistent gain. [Measurements](p6p/trials.txt), [local bitonic sorting implementation](../ggml/src/ggml-cuda/argsort.cu) and [Batcher, Sorting networks and their applications](https://doi.org/10.1145/1468075.1468121)
- A cooperative grid executed Q2_K gate/up, SITU with Q8_1 quantization and Q2_0 down in one launch for each of 44 visits. The other four use IQ4_NL. The prototype matched all 64 reference logit rows. Removing 88 launches cost time. The fused grid took 34.99 us/visit versus 19.37 + 1.83 + 12.02 = 33.22 us for separate launches. Using 4, 8 or 16 warps per block, prefetching down weights, unrolling output rows and replacing grid barriers with per-expert readiness flags all lost to the two-row down kernel. [Resident trace](p6p/resident.profile), [matched original trace](p6p/original.profile), [timings](p6p/trials.txt), [cooperative launch in this backend](../ggml/src/ggml-cuda/softmax.cu) and [Hazy Research's persistent decoder](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- A separate IQ4_NL probe tested aligned 8-byte code loads with the exact FP16 scales stored separately. It rotated matrices exceeding L2 and checked identical FP32 output before timing. Evict-first hints improved the 8600 x 2304 matrix from 16.36 to 16.02 us, did little for the 2304 x 3072 matrix, and slowed the head from 300.28 to 302.21 us. These are standalone measurements. The model keeps its GGUF layout. [Timing by layout](p6p/dense-layout.txt), [existing IQ4_NL dot](../ggml/src/ggml-cuda/vecdotq.cuh) and [NVIDIA PTX cache eviction priorities](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints)

## The remaining gap to 300 tok/s

As a diagnostic, we replaced the dense, expert and head matvecs with volatile vector loads of the selected weights and zero-filled the outputs. A node-level trace confirmed 193 replacement launches per token and zero original quantized matvec launches. The router, attention, controller and remaining elementwise kernels still ran. This took 4.646 ms/token (215.26 tok/s). With the same zero outputs and no weight reads it took 2.219 ms/token (450.69 tok/s). Synthetic activations change routing, so these two numbers describe the diagnostic schedule only. [Control patch](p6p/stream-control.patch), [measurements](p6p/trials.txt) and [PTX load semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)

The 193 matvecs address 2.007 GB/token. Streaming that amount at the measured 889 GB/s takes 2.26 ms. Adding the 2.22 ms zero-read schedule gives an optimistic serial estimate of 4.48 ms, or about 223 tok/s. To fit 3.33 ms, a different schedule would need to hide or remove roughly 1.14 ms even under those assumptions. The persistent expert grid and attention/output-weight overlap experiments both ran slower. These measurements bound the benefit available from improving the serial weight kernels. The absolute hardware ceiling with a different decoder is unknown. Server-side host overlap does not enter the timed path. The tg128 loop generates its input token locally and synchronizes after each decode. [Read bandwidth](p6p/bandwidth.txt) and [llama-bench test_gen](../tools/llama-bench/llama-bench.cpp)

A second diagnostic used the final attention and router. A side stream read one contiguous 2.007 GB buffer of initialized data per token, concurrently scheduled with the zero-output decoder. It assumed all expert choices and addresses were available before the token started, and did no dot products. The CUDA event at the end required both streams to finish. All 385 forward passes (one warm-up plus three runs of 128) had one read each. Serializing that read took 4.30 ms (232.74 tok/s), versus 2.07 ms for the decoder alone. Side-stream grids of 82, 164, 328, 656 and 1312 blocks reached 233.90, 234.00, 233.65, 243.31 and 236.03 tok/s. The fastest separate-stream schedule took 4.11 ms. Nsight verified that the two streams ran concurrently: over 119 warmed tokens, the read averaged 2.277 ms and overlapped other GPU kernels for 0.626 ms/token. This diagnostic omits model arithmetic and cannot establish the limit for a single persistent grid. To replay it on a disposable copy of the measured revision, apply the serial patch and then the overlap patch, rebuild and run with IK_DIAG_STREAM=2 and IK_IDEAL_BLOCKS=656 (negative for serialized reads). [Control patch](p6p/stream-control.patch), [overlap patch](p6p/ideal-overlap.patch), [raw runs](p6p/ideal-overlap.txt) and [NVIDIA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

The real decoder also repeated tg128 five times with CUDA graph chunk sizes 512, 0 (single graph), 1024, 256 and 512 in that order. The means were 215.57, 215.21, 215.14, 214.54 and 214.57 tok/s. The default remains 512 nodes. [Graph capture implementation](../ggml/src/ggml-cuda/ggml-cuda.cu) and [earlier phase-1 graph experiment](RESULTS.md)

## Reproduction

The measured code is commit 6d600fc on inkling-3090-plain. All selected paths are enabled for their measured GA10x shapes. The attention specialization requires single-token decode and is disabled under IK_MR. To reproduce the same-binary control, set IK_Q2_DOWN_ROWS=0, IK_ATTN_FIXED=0 and IK_ROUTER_WARP_TOPK=0 together. Setting only IK_ATTN_SHORT=0 retains the shape specialization and restores the previous short-context scan.

```bash
. perf/p5env.sh
export CUDA_VISIBLE_DEVICES=0 GPU=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$PWD/build/bin"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER="$HOME/opt/cuda-12.8/bin/nvcc" -DLLAMA_CURL=OFF
cmake --build build -j6
build/bin/llama-bench -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768 -r 3
build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 2048 -b 2048 --chunks 8
build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 1024 -b 1024 -ub 1 --chunks 2
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_ID,INKLING_ROUTER,INKLING_ATTN
```
