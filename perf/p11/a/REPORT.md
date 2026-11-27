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

## Native output head pilot

Five 128-token runs per leg measured 240.26 / 246.97 / 247.04 / 242.34 tok/s at zero depth. On the 1,025-token edit, the three-repeat medians were 350.2 / 357.4 / 355.1 / 354.5 tok/s. All 12 edits matched the independent p9 plain message and every emitted ID. The output projection keeps whole weight rows on each rank, so the partial dot products within a row retain their original order. [Raw paired results](native-pilot.txt).

The 128K p9 vision control completed a 128,064-token prompt and 3,000 generated tokens in both modes. The full replies matched. Peak board memory was 16,351 MiB in plain mode and 18,301 MiB with the drafter on GPU1. A 262,144-token DSpark slot reached its health endpoint but logged insufficient GPU memory for device-side features and the projector workspace. Capacity below is based on completed requests.

## Qualification at 04:55 EST

With the unrelated service gone, the one-card p9 vision server completed a 250,944-token prompt and generated 3,000 tokens in a 253,952-token slot at the original 512 microbatch. Its complete reply and all IDs matched independent p9 plain. Both p9 and the clean direct-head candidate reported allocation failures at the next tested slot, 262,144. [p9 request](capacity/quiet-p9-253952.timings), [p9 memory](capacity/quiet-p9-253952.memory), [candidate allocation failure](capacity/quiet-head-262144.oom).

The clean candidate matched all 128 interleaved full-vocabulary rows on the default one-card path and passed 64/64 targeted IQ4 backend cases. Its uncensored prefill PPL was 3.0571, identical to p9. A fresh two-card trace timed the primary head at 151.932 us, the peer head at 150.578 us and the activation copy at 5.121 us. [Backend result](checks/release-ops-head.txt), [uncensored PPL](checks/release-uncensored-pp.txt), [kernel profile](direct-profile-after.tsv). The complete ABBA and release-matrix audits are still running.
