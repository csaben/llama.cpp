# Phase 11, Stream A

## Measurements in progress

The uncensored and calibrated 2-bit models differ in 217 of 704 weight tensors. Their output heads have identical SHA-256. All placement comparisons use the uncensored model with the p9 snapshot as the reference. The 3.0530 / 4.1979 / 4.2038 historical PPL values refer to the calibrated model. [Weight comparison](model-comparison.txt).

| Short edit, 384 tokens | p9 before | Candidate 1 | Candidate 2 | p9 after |
|---|---:|---:|---:|---:|
| Target restricted to GPU1, drafter on GPU0 | 339.1 | 285.7 | 285.0 | 343.0 |
| Both backends available to target, target weights on GPU1, drafter on GPU0 | 351.4 | 307.9 | 302.1 | 337.0 |

Each cell is the median of three uncached fixed-length greedy completions. All repeats matched the one-card plain reference on every token ID and the full message. Registering both backends allows the target to consume the draft's device rows. The draft still needs the last committed feature and the verifier needs its proposed IDs. With both backends registered, the load log places 9,130.59 MiB of target weights and both target KV buffers on physical GPU1, and 474.81 MiB of draft weights and both draft KV buffers on physical GPU0. [Paired logs and hashes](placement.txt).

The built-in tensor split rejected the inkling2_loop architecture and row mode reported no CUDA split buffers. The exchange-only graph takes 13.1-13.3 us for one dependent 9 KiB round trip with 256-thread copy blocks. With one block per transfer, a single initial fork and a final join, system-scope flags cut 15.9-16.1 us to 7.1 us. Both ranks checked the full payload. [Budget](BUDGET.md), [graph flags](graph-flags.txt).

## Sources

- [P2P kernel module used on this machine](https://github.com/aikitoria/open-gpu-kernel-modules/tree/0d96f83bbb4669b17f5978c3d5eaef23f3e5415f).
- [gpu-unlock revision used on this machine](https://github.com/zagitta/gpu-unlock/tree/7eb51bccae0a0b7e599e084d971170e9a1e82d0c).
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html), [Phase 9 verification](../../p9/verify.sh), [Phase 10 benchmark definitions](../../PHASE10_NUMBERS.md).
