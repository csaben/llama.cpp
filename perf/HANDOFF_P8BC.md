# Phase 8 B/C handoff

Branch `inkling-3090-p8bc`, starting at brief `ad9971b`. GPU 0 is the RTX 3090 at PCI 01:00.0. Source `perf/p8bc/env.sh` from this tree before running a probe. The previous launchers re-source the GPU 1 environment internally.

Stream B started with the frozen Phase 7c binary and its libraries, the original target and folded 32K shortlist, and the release flags. Phase 7c already uses a 12 or 16 token suffix for exact history drafting. Phase 8 compares against that hybrid baseline. The new agent turn returns a file after a tool response and user follow-up. [Release evidence](RESULTS.md#phase-7c), [runner](p8bc/probe.sh), [request](spec_probe.py).

Stream C is pending. The local build uses the Phase 7c source and the Release, CUDA 12.8, SM86 and curl flags recorded in [the build instructions](RESULTS.md).

Frozen GPU 0 plain medians at 384 tokens were 208.5 code, 198.7 edit, 208.1 prose, 180.3 edit@16K, 166.2 edit@32K and 191.8 agent. The first cap-3 controls were 241.9, 327.7 and 209.1 on code/edit/prose. Reply-dependent history acceptance is being tested with a bounded longest-suffix search, periodic continuations and conditional success counts through the first rejected token. [Baseline captures](p8bc/baseline/384/).

The frozen release now has complete GPU 0 controls for all six workloads and both budgets. All comparisons passed, including the five cross-GPU plain references. Agent cap 3 measured 280.0 / 308.8 tok/s at 384 / 3,000 tokens. [Table and captures](RESULTS_P8BC.md#frozen-gpu-0-baseline).
