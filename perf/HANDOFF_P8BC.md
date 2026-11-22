# Phase 8 B/C handoff

Branch `inkling-3090-p8bc`, starting at brief `ad9971b`. GPU 0 is the RTX 3090 at PCI 01:00.0. Source `perf/p8bc/env.sh` from this tree before running a probe. The previous launchers re-source the GPU 1 environment internally.

Stream B started with the frozen Phase 7c binary and its libraries, the original target and folded 32K shortlist, and the release flags. Phase 7c already uses a 12 or 16 token suffix for exact history drafting. Phase 8 compares against that hybrid baseline. The new agent turn returns a file after a tool response and user follow-up. [Release evidence](RESULTS.md#phase-7c), [runner](p8bc/probe.sh), [request](spec_probe.py).

Stream C is pending. The local build uses the Phase 7c source and the Release, CUDA 12.8, SM86 and curl flags recorded in [the build instructions](RESULTS.md).
