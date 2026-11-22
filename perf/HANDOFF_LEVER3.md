# Phase 7 GPU0 handoff

Branch inkling-3090-lever3. The executable source at cdef6ff equals 16264c6 (only the brief differs). GPU0 is the RTX 3090 at PCI 01:00.0, UUID GPU-a5900811-5e3c-f130-cec4-7b2867cd8203. The local Release build uses CUDA 12.8, sm_86, CUDA graphs and the phase 6 plain build options.

Baseline build is in progress. Measure the four depths with the calibrated Q2 file and collect a node trace. Experiments will target overlap in the graph, then an expert activation and down-projection instruction within one visit. Keep all experiments opt-in while screening. Source perf/p5env.sh before setting CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0, GPU=0 and LD_LIBRARY_PATH=$PWD/build/bin. The unmodified r4_checks.sh selects GPU1 internally, so run its mr_check commands with this environment.

The fresh baseline reached 217.75, 198.50, 185.95 and 170.84 tok/s at depths 0, 4096, 16384 and 32768 (three repetitions). The Nsight summary and raw benchmark table are in perf/p7l3. The 300 tok/s target needs 1.259 ms/token off the unprofiled 4.592 ms baseline. The CUDA baseline library is preserved in build/p7l3/base-libs.
