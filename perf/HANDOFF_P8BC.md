# Phase 8 B/C handoff

Branch `inkling-3090-p8bc`, starting at brief `ad9971b`. GPU 0 is the RTX 3090 at PCI 01:00.0. Source `perf/p8bc/env.sh` from this tree before running a probe. The previous launchers re-source the GPU 1 environment internally.

Stream B started with the frozen Phase 7c binary and its libraries, the original target and folded 32K shortlist, and the release flags. Phase 7c already uses a 12 or 16 token suffix for exact history drafting. Phase 8 compares against that hybrid baseline. The new agent turn returns a file after a tool response and user follow-up. [Release evidence](RESULTS.md#phase-7c), [runner](p8bc/probe.sh), [request](spec_probe.py).

Stream C is pending. The local build uses the Phase 7c source and the Release, CUDA 12.8, SM86 and curl flags recorded in [the build instructions](RESULTS.md).

Frozen GPU 0 plain medians at 384 tokens were 208.5 code, 198.7 edit, 208.1 prose, 180.3 edit@16K, 166.2 edit@32K and 191.8 agent. The first cap-3 controls were 241.9, 327.7 and 209.1 on code/edit/prose. Reply-dependent history acceptance is being tested with a bounded longest-suffix search, periodic continuations and conditional success counts through the first rejected token. [Baseline captures](p8bc/baseline/384/).

The frozen release now has complete GPU 0 controls for all six workloads and both budgets. All comparisons passed, including the five cross-GPU plain references. Agent cap 3 measured 280.0 / 308.8 tok/s at 384 / 3,000 tokens. [Table and captures](RESULTS_P8BC.md#frozen-gpu-0-baseline).

The first suffix trial matched all 12 short workload/cap pairs and their repeats. At cap 3 it measured 242.3 code, 334.0 edit, 205.3 prose and 280.2 agent. Prose history accepted only 10-12 of 25-28 proposed tokens per request. At cap 6 the selector mostly used width 4 on copies, and short edits slowed despite higher total acceptance. A second trial requires five matching tokens and measures each lookup width on the request. [Trial output](p8bc/trials/three-token-min.txt).

The dense IQ4_NL and segmented projections now also specialize five, six and seven target rows, preserving the scalar partial-sum order. Against the original library on GPU 0, those windows fell from 13.78 / 15.84 / 17.56 to 13.10 / 15.17 / 16.82 ms at prefix 500. Both runs matched all 880 full-vocabulary rows. Four rows measured 10.91 and 10.94 ms. [Paired window evidence](p8bc/windows/).

The selected policy adds a confidence adjustment when DSpark and token history agree. In a five-repeat short trial, cap 6 measured 244.0 / 338.6 / 214.1 / 290.5 tok/s on code / edit / prose / agent. The four short and four long messages and all IDs matched frozen plain. The full release matrices, rollback and CUDA batteries are in progress. [Ranked trials](RESULTS_P8BC.md#draft-selection-trials).

The release-flag rollback/boundary battery matched all 2,714 rows, all six CLI greedy probes reproduced Phase 7c, and CUDA passed 1,108/1,108 selected cases. Prefill PPL reproduced 3.0530 and decode PPL 4.1979 / 4.2038. The default battery is running. Use `CHECKS_GPU=0` when invoking the legacy checker. [Release checks](p8bc/checks/).

Both correctness batteries are complete. The eight additional depth tests matched all 1,456 kept rows, the bytewise frozen comparisons passed at all depths and with interleaved sequences, and the two synchronization checks and 4,229 expanded CUDA cases passed. The full served matrix started after these checks. [Verifier results](RESULTS_P8BC.md#stream-b-verifier-checks).

The 384-token main matrix passed all 42 drafted cells (three identical repeats each) against a fresh plain server. That includes guarded and explicit adaptive n2/n3/n4 and guarded n6 on all six prompts. Additional prose, forced-width controls and the 3,000-token matrices are running serially on GPU 0. [Main 384-token measurements](p8bc/matrix/main-384.txt).

The additional short prose suite passed every byte and ID in all 100 drafted requests. The last review/dialogue cells ran below the initial plain control. GPU 0 reached 86 C and reported changing clocks during these rows. Paired runs bracket both implementations and repeat the same prompt order. [Additional short prose](p8bc/matrix/prose-384.txt).

All 280 short drafted requests (107,520 emitted IDs) now match same-flags plain, including the forced-width controls. The 3,000-token matrix is in progress. [Fixed-width 384](p8bc/matrix/fixed-384.txt).

All 42 guarded 3,000-token main cells match the full plain messages and 126,000 IDs. The cap-6 agent cell measured 270.9 tok/s after the long-context cells in this matrix. Rates depend on thermal state and prompt order, so the paired comparison remains the throughput reference. [Long main matrix](p8bc/matrix/main-3000.txt).

Both 7c matrices are complete, including the new six-token setting and agent workload. All 360 drafted messages and 347,520 IDs matched. The frozen/local crossover is running with identical prompt ordering and GPU clock samples. [Complete matrices and cap-6 table](RESULTS_P8BC.md#complete-phase-7c-request-matrices-on-gpu-0).
