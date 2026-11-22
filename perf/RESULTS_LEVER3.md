# Phase 7 GPU0 plain decode

[Round 3 dependency measurements and results](#round-3-on-gpu0) | [Round 2 streaming floor](#round-2-on-gpu0)

## Plain single-token decode

The selected build reached **219.23 tok/s at depth zero** on GPU0. Its mean also rose at 4K, 16K and 32K. The default fold replaces the attention output projection and its recurrent four-tap convolution at all 48 visits. Nsight measured 6.788 us per fused call versus 8.614 us for the original pair, saving about 0.088 ms/token across those visits. The complete profiled interval fell from 4.939 to 4.829 ms. The real decoder still needs to shed 1.228 ms/token to reach 300 tok/s. [Baseline trace](p7l3/base-trace.txt), [selected trace](p7l3/accepted-trace.txt)

| Cache depth | Fresh baseline, tok/s | Selected fold, tok/s | Change |
|---:|---:|---:|---:|
| 0 | 217.75 +/- 3.61 | 219.23 +/- 2.76 | +0.68% |
| 4096 | 198.50 +/- 4.32 | 200.86 +/- 3.36 | +1.19% |
| 16384 | 185.95 +/- 2.61 | 187.12 +/- 1.76 | +0.63% |
| 32768 | 170.84 +/- 0.86 | 173.51 +/- 1.59 | +1.56% |

Each entry is the mean and sample standard deviation of three tg128 repetitions. [Fresh baseline](p7l3/baseline.txt), [selected build](p7l3/final.txt)

The same binary also ran off/on/on/off in four consecutive sweeps. Averaging the two means per setting gives six repetitions per depth. The initial and terminal off sweeps differed by up to 2.11 tok/s, so the same-binary comparison is important for interpreting a small gain. [First off](p7l3/paired-off-a.txt), [first on](p7l3/paired-on-b.txt), [second on](p7l3/paired-on-c.txt), [last off](p7l3/paired-off-d.txt)

| Cache depth | Fold off, tok/s | Fold on, tok/s | Change |
|---:|---:|---:|---:|
| 0 | 212.86 | 215.56 | +1.27% |
| 4096 | 194.45 | 197.41 | +1.52% |
| 16384 | 183.02 | 186.76 | +2.05% |
| 32768 | 169.56 | 172.49 | +1.73% |

## Correctness

| GPU0 quality check | Fresh baseline | Selected fold |
|---|---:|---:|
| Prefill perplexity, 8 x 2048 | 3.0530 | 3.0530 |
| Decode perplexity, 2 x 1024, ubatch 1 | 4.1979 | 4.1979 |

The baseline quality run loaded the preserved baseline GGML libraries. After a 200-token corpus prefix, all 64 single-token rows of 200058 FP32 logits were byte-identical. The trace replay compared 1699 DSpark output rows. Short windows, mixed widths, rejects and rollbacks added 1135 comparisons at prefixes through 4500, including the reference attention control. All 2834 rows matched single-token decoding bit-for-bit, with maximum absolute logit difference zero. [PPL output](p7l3/ppl.txt), [logit hashes](p7l3/base-final-hashes.txt), [DSpark rows](p7l3/mr.txt), [attention control](p7l3/mr-control.txt)

The complete sm_86 Release build passed. CUDA0 passed all 3988 supported backend cases selected here, including 2696 MUL_MAT, 1081 MUL_MAT_ID, 162 attention/router/convolution/residual/controller cases, five SITU cases and 44 state-output cases. The state tests concatenate the updated state with the operator output so both are checked. The harness lists unsupported shapes separately. [Operator summary](p7l3/backend.txt), [SITU](p7l3/backend-situ.txt), [state checks](p7l3/backend-state.txt), [operator fixtures](../tests/test-backend-ops.cpp)

## What limits the measured decoder

On the fresh baseline, 300 tok/s requires saving 1.259 ms from a 4.592 ms token. Even granting all 193 weight matvecs their inputs and expert choices at the start of each token, the best concurrent replay took 4.091 ms. The two resident kernels also lost at the individual layer visits where scaling would have to begin. These results rule out the tested schedules as a route to 300 tok/s on this card. A different decoder or weight layout could have a different limit. [Phase 7 brief](PHASE7_LEVER3_BRIEF.md), [fresh baseline](p7l3/baseline.txt), [replay measurements](p7l3/oracle.txt)

### Dependency-free replay

The diagnostic used the model's actual dense, expert and vocabulary matrices in decoder order. One graph branch ran the stock matvecs on frozen Q8_1 inputs and eight predetermined expert IDs per routed visit. A second branch ran the real glue kernels with zero stand-ins for the matvec outputs. Both branches had to finish before the next iteration. This lets the scheduler see future weight addresses earlier than a real decoder and changes the data seen by the kernels. Its rates measure synthetic iterations. The previous plain round used a contiguous read in place of the matvecs and reached 243.31 iterations/s. [Two-branch results](p7l3/oracle.txt), [Phase 6 read-only experiment](p6p/ideal-overlap.txt), [CUDA graphs and asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

| Graph schedule | First run, iterations/s | Repeat, iterations/s |
|---|---:|---:|
| 193 matvecs alone | 332.38 | 330.04 |
| Glue alone | 548.13 | 549.29 |
| Both branches serialized | 216.94 | 216.20 |
| Both branches concurrent | 244.44 | 244.28 |

With native graph chunking, the matvecs alone took 3.027 ms (330.36/s). That leaves about 0.306 ms in the target budget for every exposed dependency, including attention, routing, recurrent state and dispatch. In a separate instrumented concurrent run, Nsight counted 193 matvec calls and 441 glue calls per token, with 3.149 ms of matvec activity, 2.209 ms of glue activity and 1.263 ms when both streams were active. The profiled head-to-head interval was 4.556 ms. Timing conclusions above use the unprofiled runs. [Matvec-only measurement](p7l3/matvec-only.txt), [kernel counts](p7l3/matvec-only-trace.txt), [overlap](p7l3/oracle-overlap.txt), [interval accounting](p7l3/branches.py)

## Experiments on the real decoder

| Experiment | Depth-zero result or per-visit comparison | Local tag |
|---|---|---|
| Graph weight prefetch in nine glue gaps | All gaps at 256, 1024 and 3072 KiB reached 210.58, 204.34 and 198.71 tok/s. Controls ranged from 212 to 216. | `p7l3-prefetch-204tok` |
| Inline weight prefetch and L2 retention | Three inline sites combined reached 213.90, 212.54 and 208.83 tok/s at 128, 512 and 2048 KiB. Evict-last also lost. | `p7l3-inline-prefetch-213tok` |
| Resident SITU / Q2_0 expert down | 44 visits/token, 18.65 us vs 11.85 us for the existing pair. Best screen about 202 tok/s. | `p7l3-resident-202tok` |
| Resident attention / IQ4_NL output projection | 48 visits/token, 19.92 us vs 15.58 us for the original pair. Per-head readiness took 26.70 us. | `p7l3-attention-visit-212tok` |
| Q2_K expert gate/up with SITU and Q8 packing | Best of four warp counts reached 212.11 tok/s vs 215.63 and 216.29 controls. | `p7l3-gate-up-212tok` |
| Controller apply with RMS and Q8 packing | 24 visits/token, 6.78 us. Off/on/on/off reached 216.75/215.92/215.22/215.43 tok/s. | `p7l3-controller-216tok` |
| IQ4_NL output projection with four-tap short convolution | One row per warp reached 219.95 and 218.59 tok/s vs 217.28 and 215.48 controls. | `p7l3-projection-fold-220tok`, `p7l3-fold-tiles-219tok` |

**Weight prefetch.** The graph mask selects residual mix (1), controller down (2), controller apply (4), attention (8), short convolution (16), router (32), SITU (64), MoE sum/convolution (128) and standalone normalization (256). Each gap ran separately at three chunk sizes. During routing, the selected experts are unknown, so the producer sampled prefixes across possible experts. SITU already has all eight IDs and can prefetch exactly the selected down weights. Nsight counted 412 prefetch launches and 0.602 ms of prefetch activity per token across all gaps. Overlap occurred, but the interval rose from 4.939 to 5.148 ms. Testing evict-last with different warp counts, and inserting extra CTAs directly into three existing kernels, also failed to improve throughput. [Nine-gap sweep](p7l3/gaps.txt), [per-gap overlap](p7l3/gap-trace.txt), [retention](p7l3/retain.txt), [inline sweep](p7l3/inline.txt), [inline trace](p7l3/inline-trace.txt), [NVIDIA PTX prefetch and cache hints](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)

**Resident execution.** The expert grid packs SITU into Q8_1 per expert slot. Workers double-buffer Q2_0 tiles with asynchronous global-to-shared copies. They can wait for one slot when the allocation permits it, or for all producers before overwriting aliased inputs. For attention, eight head-group producers publish the packed activation while resident workers take strided output-row tiles. Trials varied direct loads, L2 hints, shared-memory staging, tile size and worker count. Per-head flags let a worker compute partials before all heads were ready. The fastest direct-load screen reached 212.36 tok/s. Neither pipeline beat the stock visit, so the brief's performance gate stopped expansion to a whole-token grid. [Expert trials](p7l3/resident.txt), [expert trace](p7l3/both-trace.txt), [attention trials](p7l3/visit.txt), [per-head and corrected tile runs](p7l3/visit-heads.txt), [staged trace](p7l3/visit-trace.txt), [per-head trace](p7l3/heads-trace.txt), [Hazy Research megakernel design](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [NVIDIA asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)

**Glue folding.** The Q2_K gate/up epilogue produced the packed Q8_1 activation for expert down, but its smaller tiles slowed the weight multiply. The controller variant reproduced the original reduction across 32-channel groups. The useful fold was the IQ4_NL output projection. Its writer applies the four taps, residual and scale, then stores the updated recurrent state directly. A CUDA allocation dependency keeps the packed input live through the write. The first prototype hit only 21 of 48 visits because the graph allocator reused that storage. A numeric probe also found that reproducing Ampere MMVQ's four fused multiply-add accumulators, reduction order and writer lane was necessary for bitwise equality. One- and two-row warp layouts and CTA reductions were timed. [Gate/up sweep](p7l3/up.txt), [controller sweep](p7l3/ctrl.txt), [partial-match timings](p7l3/fold-ab.txt), [48-visit timings](p7l3/fold-pin.txt), [CTA timings](p7l3/cta.txt), [state and output comparisons](p7l3/fold-exact.txt), [MMVQ](../ggml/src/ggml-cuda/mmvq.cu), [IQ4_NL dot](../ggml/src/ggml-cuda/vecdotq.cuh), [reference short convolution](../ggml/src/ggml-cuda/inkling.cu)

## Reproduction

All timings used the calibrated 2-bit Inkling 2 Mini GGUF selected by Q2, a single RTX 3090 at PCI 01:00.0 (82 SMs), CUDA 12.8, a Release sm_86 build, full offload, flash attention and Q8_0 K/V. The GGUF mixes quantization types. llama-bench labels it IQ4_NL. The baseline checkout was cdef6ff, whose only change from 16264c6 is the Phase 7 brief. The local experiment tags identify the source for each trial. The build identifier in interim llama-bench output reflects the last CMake configuration. The previous plain worktree supplied the build settings and trace methodology. [Previous plain results](RESULTS_PLAIN.md), [plain handoff](HANDOFF_PLAIN.md), [plain brief](PHASE6_PLAIN_BRIEF.md), [Phase 6 DSpark](RESULTS.md#round-4---long-reply-exactness-gpu1-rtx-3090), [benchmark driver](../tools/llama-bench/llama-bench.cpp)

```bash
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0
export LD_LIBRARY_PATH="$PWD/build/bin"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER="$HOME/opt/cuda-12.8/bin/nvcc" -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build -j8
build/bin/llama-bench -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768 -r 3
build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 2048 -b 2048 --chunks 8
build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 1024 -b 1024 -ub 1 --chunks 2
```

Set `IK_PROJ_CONV=0` for the same-binary control. The fold applies only to the matching single-token GA10x shape. `IK_MR=1` takes the existing verification path. Nsight Systems 2024.6.2 used CUDA node-level graph tracing, disabled sampling, SQLite export and 119 warmed head-to-head intervals. [Trace summarizer](p7l3/trace.py), [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [model graph](../src/models/inkling2.cpp), [CUDA allocation and dispatch](../ggml/src/ggml-cuda/ggml-cuda.cu), [fold implementation](../ggml/src/ggml-cuda/inkling-fmv.cu)

At the experiment tags, use the same benchmark with `-d 0`. Set `IK_GAP_MASK` and `IK_GAP_KIB` to reproduce a prefetch row. For the resident variants, set `IK_RESIDENT_DOWN=1` or `IK_ATTN_VISIT=1`. For the replay at `p7l3-oracle-244tok`, set `GGML_CUDA_GRAPH_CHUNK_SIZE=0` and `IK_DIAG_MATVECS_ONLY` to 1 (matvecs), 4 (glue), 3 (serial) or 2 (concurrent). [Graph capture and experiment history](HANDOFF_LEVER3.md), [baseline kernel pairs](p7l3/base-trace.txt)

For DSpark, build the repository's `mr_check` and export the device selection above before running it. The trace replay uses the original graph reserve of three and a 16384-token context. The other comparisons use a reserve of four. The corpus and trace are part of the repository. [Reference checker](p6/mr_check.cpp), [Round 4 harness](p6/r4_checks.sh), [captured replay](p6/r4_repro/n3.trace.gz), [corpus](ppl.txt)

```bash
g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p6/mr_check.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o build/bin/mr_check
(
    export IK_MR=1 IK_MMA=3 MR_RS_A=0 MR_RS=4
    gzip -dc perf/p6/r4_repro/n3.trace.gz > build/p7l3-r4.trace
    MR_RS=3 MR_CTX=16384 MR_TRACE="$PWD/build/p7l3-r4.trace" build/bin/mr_check "$Q2" perf/ppl.txt
    build/bin/mr_check "$Q2" perf/ppl.txt 300 24
    for n in 1533 3069 4500; do
        MR_RAND=2 MR_SIZES=5 MR_PLAIN_PCT=0 build/bin/mr_check "$Q2" perf/ppl.txt "$n" 24
    done
    IK_ATTN_FIXED_MR=0 MR_RAND=2 MR_SIZES=5 build/bin/mr_check "$Q2" perf/ppl.txt 3069 24
    for n in 2000 2046 2558 3069 4094 4500; do
        MR_RAND=37 MR_SIZES=2,3,4,5 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 build/bin/mr_check "$Q2" perf/ppl.txt "$n" 64
    done
)
```

The logit dumper uses the context and corpus setup of the previous plain round's local `build/p6p/logits.cpp`, derived from the window checker. The archived version reproduced the baseline SHA-256 on the final library. [Logit dumper](p7l3/logits.cpp), [previous validation](p6p/validation.txt), [window checker](p6/mr_check.cpp)

```bash
g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p7l3/logits.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o build/bin/p7-logits
build/bin/p7-logits "$Q2" perf/ppl.txt > build/p7-logits.bin
sha256sum build/p7-logits.bin
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID
build/bin/test-backend-ops test -b CUDA0 -o INKLING_ATTN,INKLING_ROUTER,SHORTCONV_RES,RES_MIX,LOOP_CTRL_DOWN,LOOP_CTRL_APPLY
build/bin/test-backend-ops test -b CUDA0 -o SITU
build/bin/test-backend-ops test -b CUDA0 -o CONCAT -p 'n_slots=|n_used='
```

## Round 2 on GPU0

The final default build reached **223.07, 205.20, 192.18 and 177.46 tok/s** at depths 0, 4096, 16384 and 32768. The isolated production-order dot kernel took **2.775 ms**. Real decode took **4.483 ms** at depth zero, 1.150 ms over the 300 tok/s budget. The accepted change overlaps the late gate with the attention scan. A single resident kernel through the router lost its performance advantage. Selected source `a4b8d65`, final local tag `p7l3r2-final-223tok`. [Default benchmark](p7l3r2/default.txt), [actual baseline](p7l3r2/official-base.txt), [bitwise floor](p7l3r2/bitwise-floor.txt), [resident extension](p7l3r2/resident-router-tune-trace.txt)

### Streaming floor for one real route

One resident kernel evaluated the **193 quantized matvecs in 2.692 ms**. A CUDA graph with the 193 production matvec launches took **2.871 ms** before it and **2.892 ms** after it. The measured difference was **0.179-0.200 ms/token**. The 300 tok/s target allows 3.333 ms for everything, leaving 0.642 ms after the resident dot workload for unhidden attention, routing, controllers, recurrent state and other work. [Raw event batches](p7l3r2/floor-first.txt), [launch sweep](p7l3r2/floor-sweep.txt), [brief](PHASE7_LEVER3_BRIEF.md#round-2-after-tag-p7l3-final-219tok)

| Schedule on the captured weights | Median ms/token | Logical weight GB/s |
|---|---:|---:|
| Single resident kernel, real quantized dots | 2.692 | 745.45 |
| Production matvecs in a 193-node graph, before/after | 2.871 / 2.892 | 698.9 / 693.9 |
| Same row-dot implementation as the resident kernel, 193-node graph | 3.527 | 568.9 |
| Single resident kernel, volatile reads | 2.340 | 857.4 |

After a 200-token corpus prefix, the probe took the weight pointers from the live decode graph. Immediately before each routed matvec consumed its IDs, it copied the eight actual selections. Each of 48 gate/up and down pairs agreed, with eight distinct IDs out of 128. The descriptor list preserves visit, matrix, expert slot and row order, including the separate Q/K/V input columns. The timed kernels use the same deterministic nonzero Q8_1 activations. They write every consumed output row. The tiled dot changes accumulation order. All 2,494,400 rows agreed with the production kernels within tolerance, with maximum absolute difference 4.77e-7. [Full matrix inventory and routes](p7l3r2/routes.txt), [probe](p7l3r2/floor.cuh), [corpus driver](p7l3/logits.cpp), [native MMVQ](../ggml/src/ggml-cuda/mmvq.cu), [quantized dot implementations](../ggml/src/ggml-cuda/vecdotq.cuh)

The numerator is **2,006,498,304 bytes** per token. The 96 dense matrices contribute 726,755,328, the 96 selected expert matrices 1,019,215,872, and the vocabulary head 260,527,104. This counts visits to reused layers. The separate F32 router and F16 controller weights add about 169.87 MB to real decode. The brief's 2.42 GB estimate also differs from the sum of its listed components. The bandwidth column uses logical weight bytes. Hardware traffic also includes instructions' other operands, outputs, repeated transactions and cache effects. [Captured weights](p7l3r2/routes.txt), [prior GGUF inventory and card bandwidth measurement](RESULTS_PLAIN.md)

The 82-SM GA102 grid had 820 CTAs, 128 threads each, and 155,924 row tiles of up to 16 rows. Each CTA advances by the grid size across all matrix boundaries. A matrix has no separate launch, synchronization, or final underfilled wave. There is one initial fill and one final drain for the entire token. Each reported value is the median of five CUDA-event batches of 64 invocations after 16 warmups. Both graph controls verified exactly 193 nodes. The volatile PTX control reads the full byte payload. The three-route audit repeated the comparison after 0, 12 and 48 decode steps, checked the distinct route hashes, and validated all output rows. A later layout-capable build used nine CTAs/SM and took 2.749-2.805 ms on the native layout. The last packed run had a slow final graph control, visible in the raw summary. [Launch sweep](p7l3r2/floor-sweep.txt), [matched route audit](p7l3r2/layout-floor.txt), [audit script](p7l3r2/summarize_floor.py), [CUDA graphs, occupancy and event timing](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html), [PTX loads, barriers and memory ordering](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

A second resident kernel reproduced the production warp accumulators and writer lane. **All 2,494,400 FP32 results matched bit-for-bit** in **2.775 ms**, compared with 2.935/2.948 ms for the native graph and 3.735 ms for its own 193-node graph. An untimed instrumented dispatch read the hardware SM ID in every block. Exactly ten CTAs ran on each of the 82 SMs. This leaves 0.558 ms of the 300 tok/s budget with production rounding. The measured native-graph savings were 0.160-0.174 ms for this run. Local source tag `p7l3r2-bitwise-floor`. [All SMs, bitwise checks and event batches](p7l3r2/bitwise-floor.txt), [exact dot and graph controls](p7l3r2/floor.cuh)

### From resident tiles to real decode

| Experiment | Measured GPU0 result | Local source tag |
|---|---|---|
| Per-head resident instruction lists | A single cooperative launch executed QKV, Q8 attention, WO and short convolution including state at all 48 visits. It took 37.779 us/visit against 31.789 us for the graph. | `p7l3r2-dense-visit-v1` |
| Producer queues and prefetch | Dedicated scan CTAs, two register budgets, L2 hints and asynchronous shared-memory staging failed to recover the dense graph cost. | `p7l3r2-late-gate-221tok` |
| Late gate in one cooperative launch | Scheduling the 3,072 gate rows after the scan inputs let the scan and gate overlap. The whole dense chain took 31.630 us against 31.839 us for the graph, all 48 visits. | `p7l3r2-late-gate-221tok` |
| Three launches | Separate non-gate rows, shared gate/scan grid, then WO+shortconv ran in 31.320 us/visit. Twelve gate CTAs/head performed best in the short sweep. | `p7l3r2-late-gate-221tok` |
| Extend through output and router | Actual router choices were computed in-kernel at 24 direct visits/token. The output+router section rose from 15.397 to 19.290 us. | `p7l3r2-visit-extensions` |
| Entire dense chain and direct router in one resident launch | At 24 visits, QKV, attention, WO, state update and top-8 router selection took 49.234 us when staging router weights in shared memory where they fit, or 49.621 us with L2 prefetch throughout, against 40.797 us for the winning single dense kernel followed by the native router. At the other 24 visits the controller intervenes. Both variants matched the full reference logits. | `p7l3r2-whole-visit-router` |
| Extend upstream through residual mix | A cooperative launch overlapped input-weight staging with mix, normalization and Q8 packing. It ran at 24 visits/token. Preloading up to 56 rows/worker cost throughput. | `p7l3r2-visit-extensions` |
| Longer cache | Separate graph streams for the gate and scan lost throughput. A 384-thread block running eight scan warps and four gate warps was faster than eight plus eight in the screen. | `p7l3r2-visit-extensions` |
| Lossless dense packing | The loader joined each layer's QKV and WO allocations and moved the same IQ4_NL scales to the end of each row. It needed 363,377,664 extra bytes. The small throughput change did not justify keeping the second layout. | `p7l3r2-dense-layout` |
| Lossless expert packing and output tiles | Packing Q2_0 codes for two fewer byte permutes per eight values took 12.397 us for down plus sum/conv against 12.168 us on native weights. A native four-row, eight-slot fused tile took 12.701 us at the 44 Q2_0 visits. | `p7l3r2-expert-layout` |

The measurements above use live router outputs, KV caches and recurrent state unless explicitly labeled as the streaming-floor experiment. Nsight intervals run from the first kernel start through the last kernel end at each visit over 119 complete token intervals. A separate full-logit run checked the dense resident and staged prototypes. The early expert screen with no eligible fusions was excluded after inspecting its trace. [First visit](p7l3r2/visit-first.txt), [queue sweep](p7l3r2/queue-screen.txt), [late-gate traces](p7l3r2/gate-traces.txt), [gate geometry](p7l3r2/stage-screen.txt), [dense logit hashes](p7l3r2/dense-hashes.txt), [router](p7l3r2/router-trace.txt), [mix](p7l3r2/mix-screen.txt), [long-cache streams](p7l3r2/long-screen.txt), [warp counts](p7l3r2/warps.txt), [dense packing](p7l3r2/pack-screen.txt), [expert screen](p7l3r2/expert-layout-screen.txt), [expert and mix coverage](p7l3r2/expert-trace.txt)

The shared-memory mode staged weights for two- and three-state routers. Four- and five-state routers used L2 prefetch to fit their per-block memory budget. A final sweep let router CTAs prefetch their own rows, execute useful dense tiles, then become router CTAs again. The best resident extension used a two-CTA/SM register budget, ran in **45.084 us** for dense plus router, and delivered **210.45 tok/s** in a screen bracketed by 216.06 and 215.70 for dense fusion alone. A three-CTA/SM register budget fell to 200.91 tok/s. All timed extensions checked for logits equality in the trace file matched the reference. This performance gate ended expansion to further token dependencies. Local source tag `p7l3r2-resident-router-tuning`. [Geometry sweep](p7l3r2/resident-router-tune.txt), [verified intervals and hashes](p7l3r2/resident-router-tune-trace.txt)

The expert epilogue initially rounded the final scale and residual add differently from the production FFN. Restoring its single fused multiply-add gave the baseline logit hash for native and packed expert tiles, and for the standalone packed matvec. The corrected native/packed full-tail timings were 12.671/12.354 us. Duplicating the 22 Q2_0 physical expert tensors would cost 2,102,132,736 bytes. Both experimental duplicates were discarded from the selected implementation. [Corrected expert traces](p7l3r2/expert-exact-trace.txt), [resident router trace](p7l3r2/resident-router-trace.txt), [resident router benchmark](p7l3r2/resident-router-screen.txt), [coverage analysis](p7l3r2/extended_trace.py)

The single-launch prototypes use a cooperative resident grid, per-task release/acquire counters, and the existing Inkling scan and quantized arithmetic. The instruction-stream design drew on Hazy Research's Llama-3.2-1B work and the Mirage Persistent Kernel paper. Mirage calls its representation an SM-level graph. Neither external implementation was linked. The experiments used local GA102 measurements for every promotion decision. [Hazy Research design](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), [Cheng et al., MPK](https://arxiv.org/abs/2512.22219), [CUDA asynchronous copies](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async), [Inkling layer graph](../src/models/inkling2.cpp), [attention](../ggml/src/ggml-cuda/inkling.cu), [projection and shortconv](../ggml/src/ggml-cuda/inkling-fmv.cu)

### Real single-token throughput and accuracy

The requested tg128 benchmark used all four depths, three repetitions, full GPU offload, flash attention and Q8_0 K/V. The baseline was a new local build of a Git archive of **16264c6**. On the final build, leaving `IK_DENSE_VISIT` unset gave the following results. Both runs started at or below 63 C on the same RTX 3090. [Baseline](p7l3r2/official-base.txt), [final default with source ID](p7l3r2/default.txt), [benchmark and verification commands](p7l3r2/validate.sh)

| Context depth | 16264c6 tok/s | Final default tok/s | Gain |
|---:|---:|---:|---:|
| 0 | 215.84 +/- 2.39 | **223.07 +/- 2.08** | +3.35% |
| 4,096 | 198.44 +/- 3.21 | **205.20 +/- 3.85** | +3.41% |
| 16,384 | 186.12 +/- 2.37 | **192.18 +/- 2.61** | +3.26% |
| 32,768 | 170.05 +/- 0.91 | **177.46 +/- 2.49** | +4.36% |

The earlier opt-in build at `p7l3r2-gate-overlap` gave 222.46, 205.37, 192.76 and 176.94 tok/s. Its generated version string names the preceding configuration commit. The final Release build identifies `a4b8d65` directly. [Opt-in run](p7l3r2/official-on.txt), [final default](p7l3r2/default.txt)

**Same-binary paired result.** At each depth four independent processes ran off/on/on/off with three repetitions each and a start temperature of 63 C. Off retains the Round 1 projection/conv fold. The table averages the two process means per condition and isolates the incremental Round 2 gain. [All sixteen runs](p7l3r2/paired.txt)

| Depth | Round 1 path, tok/s | Round 2 path, tok/s | Gain |
|---:|---:|---:|---:|
| 0 | 218.675 | 222.455 | +1.73% |
| 4,096 | 200.550 | 205.535 | +2.49% |
| 16,384 | 188.795 | 192.305 | +1.86% |
| 32,768 | 174.115 | 178.015 | +2.24% |

The selected implementation overlaps gate projection and attention, then uses the existing WO/shortconv kernel. The CUDA graph allocator keeps the Q8 input live until attention finishes. It reuses the real projection allocation for the compact scan inputs and uses the normal attention output buffer. At depth zero the 48 dense intervals averaged 30.143 us each against 31.803 us with the optimization off. At 4K all 48 visits used the long-cache warp schedule and averaged 39.112 us against 41.255 us for the graph. The final default repeated the short-context trace at 30.134 us/visit. Its depth-zero token took 4.483 ms in llama-bench, 1.150 ms over target. In the selected Nsight trace, just the router, residual mixes and two controller kernels occupied 0.976 ms per token. This already exceeds the 0.558 ms left by the bitwise dot floor. Approaching 300 tok/s requires substantially more useful overlap than the measured launch savings. [Selected implementation](../ggml/src/ggml-cuda/inkling-visit.cuh), [allocator and graph matching](../ggml/src/ggml-cuda/ggml-cuda.cu), [all kernels](p7l3r2/selected-kernels.txt), [default trace](p7l3r2/default-trace.txt), [4K intervals](p7l3r2/selected-long.txt), [paired visit traces off](p7l3r2/selected-trace-0.txt) and [on](p7l3r2/selected-trace-1.txt)

The archived baseline, opt-in build and final default all measured prefill PPL **3.0530** on 8 x 2048 and decode-path PPL **4.1979** on 2 x 1024 with `-ub 1`. All 88 teacher-forced logit rows matched byte-for-byte across the full 200,058-token vocabulary. The comparison covered 64 successive tokens after prefix 200 and eight after each prefix of 4096, 16384 and 32768 tokens. Prefix fill is batched, with the corpus token sequence repeated when needed. Five fallback configurations also matched their same-configuration controls across eight full-vocabulary rows each. [Baseline and opt-in PPL](p7l3r2/ppl.txt), [default PPL and trace replay](p7l3r2/default-quality.txt), [baseline and opt-in hashes](p7l3r2/logits.txt), [default hashes](p7l3r2/default-logits.txt), [fallback checks](p7l3r2/fallbacks.txt), [driver](p7l3/logits.cpp)

The DSpark checks compared 2,834 output rows across 12 runs, including the 1,699-row recorded trace, mixed step sizes, rollbacks, and the alternate attention control. Every comparison had maximum absolute logit difference zero. CUDA0 passed all 3,988 supported cases in the selected operator groups, including state-output fixtures for the short convolutions and expert sum. The final source enables the gate overlap automatically on the measured 82-SM GA10x shape. Set `IK_DENSE_VISIT=0` to restore the Round 1 path. Other GA10x sizes can opt in with `1`. The MR path retains its own schedule. [DSpark comparisons](p7l3r2/mr.txt), [CUDA operator groups](p7l3r2/ops.txt), [single-token graph conditions](../ggml/src/ggml-cuda/ggml-cuda.cu)

### Reproduction

Run from this worktree. The model is the 9,938,847,648-byte `Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf`, SHA-256 `bc7f51192fbc35eeb5eaec6a342028b8588691023a2399d97c29eb633a049d07`. The corpus hash is `af0be9d6d59418b9f4d599aebe17b69c55ae06bbf90b8650f6b288396d90d0be`. The paired comparison uses the same selected binary with `IK_DENSE_VISIT=0` and `1`. The experiments preserve the original prefill and multi-row paths. [Model environment](p5env.sh), [corpus](ppl.txt), [validation driver](p7l3r2/validate.sh)

```bash
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0
export LD_LIBRARY_PATH="$PWD/build/bin"
unset IK_DENSE_VISIT
build/bin/llama-bench -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768 -r 3
```

To rebuild the exact reference locally, extract `git archive 16264c6` into `build/p7l3r2/baseline-16264c6/src` and run the commands below. The validation driver's four entry points are `bench`, `quality`, `mr` and `ops`. It sets the device mask before starting CUDA programs. Full profiler databases are in `build/p7l3`, full verification logs in `build/p7l3r2`, and the small summaries in `perf/p7l3r2`. [DSpark fixtures](p6/r4_checks.sh), [backend fixtures](../tests/test-backend-ops.cpp), [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

```bash
base=build/p7l3r2/baseline-16264c6
mkdir -p "$base/src"
git archive 16264c6 | tar -x -C "$base/src"
cmake -S "$base/src" -B "$base/obj" -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$HOME/opt/cuda-12.8/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_NUMBER=0 -DLLAMA_BUILD_COMMIT=16264c6
cmake --build "$base/obj" --target llama-bench llama-perplexity -j 8
```

To reproduce the floor, extract `p7l3r2-bitwise-floor` in place of the baseline tag into a separate directory under `build/p7l3r2`. Configure its own Release sm_86 build as above and build target `llama`. Set `probe_bin` to that build's absolute `bin` directory and run the following commands. `IK_FLOOR_STEP=12` or `48` captures a different real token. The original fastest tiled floor is at `p7l3r2-floor-371iter` with the same driver and `IK_TOKEN_FLOOR=1`. The native decoder must execute while routes are captured, so leave `IK_DENSE_VISIT` and `IK_EXPERT_TAIL` off for this replay. [Floor code](p7l3r2/floor.cuh), [geometry and timing](p7l3r2/floor-sweep.txt), [bitwise control](p7l3r2/bitwise-floor.txt)

```bash
g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p7l3/logits.cpp -Iinclude -Iggml/include -L"$probe_bin" -lllama -lggml -lggml-base -o build/p7l3r2/logits-probe
LD_LIBRARY_PATH="$probe_bin" GGML_CUDA_DISABLE_GRAPHS=1 IK_DENSE_VISIT=0 IK_EXPERT_TAIL=0 IK_TOKEN_FLOOR=1 IK_FLOOR_EXACT=1 build/p7l3r2/logits-probe "$Q2" perf/ppl.txt >/dev/null
```

At `p7l3r2-late-gate-221tok`, mode 5 selects the single-launch late gate and mode 7 selects the staged short-cache schedule. Set `IK_DENSE_PREFETCH=0` and `IK_DENSE_GATE_BLOCKS=12` for the tuned short benchmark. At `p7l3r2-resident-router-tuning`, modes 18/19 select shared-memory staging or direct L2 prefetch for a resident visit through routing, and mode 22 reuses the dense workers with a two-block residency budget. At `p7l3r2-exact-expert-tail`, `IK_EXPERT_TAIL=2` selects the native-weight four-row tile and `IK_Q2_PACK=1 IK_EXPERT_TAIL=8` selects its lossless packed variant. The selected implementation uses `0` to disable the optimization and `1` to request it explicitly. [Short-context sweep](p7l3r2/stage-screen.txt), [resident sweep](p7l3r2/resident-router-tune.txt), [expert sweep](p7l3r2/expert-layout-screen.txt)

## Round 3 on GPU0

### Cost of a dependency

At 82 blocks on the RTX 3090, one point-to-point handoff took 0.352 us, 47% less than an empty CUDA graph edge. A cooperative grid barrier took 0.651 us. A single producer waking every block and waiting for acknowledgements took 0.817 us. For four dependencies per visit across 48 visits, replacing graph edges with ideal point-to-point handoffs would save 0.061 ms/token. Round 2 needed 1.150 ms/token to reach 300 tok/s. The scope of the dependency matters when projecting that saving onto decode. [Microbenchmark](p7l3r3/dependency.cu), [raw event times](p7l3r3/dependency.txt), [Round 2 rates](p7l3r2/default.txt)

| Dependent work | 2 blocks, us/edge | 16 blocks | 82 blocks | 192 edges at 82 blocks, ms |
|---|---:|---:|---:|---:|
| Empty kernels in one graph | 0.636 | 0.597 | 0.668 | 0.128 |
| Cooperative grid barrier | 0.663 | 0.652 | 0.651 | 0.125 |
| Release/acquire to the next block | 0.385 | 0.358 | 0.352 | 0.068 |
| One block publishes to all others, with acknowledgement | 0.739 | 0.798 | 0.817 | 0.157 |

Each graph has N+1 empty kernels and N edges. Each resident launch performs N handoffs. GPU events time 16 replays per sample, with seven samples per chain length and N = 32, 128, 512, 1024. Slopes use the 1024-minus-128 median divided by 896. This subtracts fixed launch and counter-reset work. The ring checks its final sequence number, and broadcast checks the total acknowledgements. All tests used 128-thread blocks. Release/acquire has device scope. The broadcast timing includes consumption by every worker before the next publication. [Implementation and final-state checks](p7l3r3/dependency.cu), [CUDA 12.8 cooperative groups and graphs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html), [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)

### What the decoder actually pays

Across 119 warmed head-to-head intervals, the top three within-token boundaries by aggregate signed idle time were router to Q2_K gate/up (42 calls, 5.688 us/token), residual mix to controller down (48, 3.958 us), and controller down to apply (48, 3.849 us). The respective mean gap per visit was 0.135, 0.082 and 0.080 us. An empty-node slope includes dispatch and block execution. Subtracting the end of one active kernel from the start of the next measures a different interval. Nsight node timestamps even overlap slightly for some serial pairs. The 332 us softcap-to-next-input gap crosses the host decode boundary. [Ranked edges and kernel durations](p7l3r3/control-edges.txt), [trace analysis](p7l3r3/edges.py)

Router work cost 350 us/token, residual mixes 285, controller-down 198, apply 147, expert sum/state update 98, separate RMS/packing 93 and SITU 89. Q2 gate/up itself averaged 19.442 us on the 42 directly adjacent router edges. Each experiment below timed useful work from the first producer's start through the last consumer's finish. Rows with different coverage use their own matched graph control. [Kernel and edge inventory](p7l3r3/control-edges.txt), [matched FFN spans](p7l3r3/matched-ffn.txt), [controller spans](p7l3r3/controller-trace.txt)

| Experiment | Matched visits/token | Graph, us/visit | Best experiment, us/visit | Local source tag |
|---|---:|---:|---:|---|
| Router and routed gate/up in one resident grid | 42 | 26.703 | 30.921 | `p7l3r3-resident-route` |
| Mix and controller down, including two fallback calls | 48 | 7.693 | 6.742 | `p7l3r3-ctrl-pair` |
| Controller down and apply | 48 | 7.272 | 7.082 | `p7l3r3-ctrl-pair` |
| Router through FFN state update | 42 | 40.585 | 62.171 | `p7l3r3-full-ffn` |
| Attention input projection through FFN state update | 19 | 71.734 | 96.101 | `p7l3r3-complete-layer` |
| Controller apply plus RMS/Q8 packing | 24 | 7.680 | 7.130 | `p7l3r3-apply-pack` |

The router/up kernel uses the 16-row Q2_K dot from the production-rounded Round 2 floor, the live normalized Q8 input and the router's actual top-eight experts. The FFN extension releases activation tiles per expert after SITU, then releases each 32-channel output tile after all eight contributions arrive. It performs the convolution and state writes in the same launch. These kernels matched all 64 vocabulary rows at each of the three tested residency settings. Two visits have a message operation between router and gate/up. Four use IQ4_NL experts. [Bit-exact floor](p7l3r2/bitwise-floor.txt), [router/up](p7l3r3/route-visit.txt), [full FFN](p7l3r3/full-visit.txt), [matched comparisons](p7l3r3/matched-ffn.txt)

The complete short-cache visit runs the per-KV-head projection instruction list, cache update and attention, output projection and recurrent convolution, router, gate/up, SITU, down and recurrent FFN update. It replaced eight graph nodes at each of 19 direct visits. It matched the 64 reference rows. The complete kernel used 128 registers/thread and actually launched 160 blocks of 256 threads at the higher requested residency. The graph could launch a separately sized grid for each stage. Higher requested residency left the complete visit at 96.10 us. Its 24.37 us loss per visit failed the brief's gate for scaling this design to a token. The Round 2 floor supplies known inputs and routes up front, so its 2.775 ms cannot serve as a live decode time. [Layer traces](p7l3r3/layer.txt), [matched spans](p7l3r3/layer-and-pass.txt), [registers and launch geometry](p7l3r3/resources.txt), [Hazy Research persistent decoder](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)

The winning mixed-state down dot recomputes the 2304-channel state mix in shared memory in each two-rank controller block. One block writes each mixed column for later users. The isolated fused calls took 6.683 us across 46 sites. Reading the message half early raised this to 7.058 us. Fusing down with apply saved less and competes for the same down operation. The apply/pack kernel instead retains 32-channel groups while 32 blocks exchange their RMS partials, then writes Q8_1 directly. The selected 256-thread version took 7.131 us. A separate router sweep loaded four expert rows in registers and formed channel products during normalization. It took 7.057 us versus 7.269 with eight shared rows. Reducing only the shared-memory tile width lost. [Mix/down](p7l3r3/mix-down.txt), [down/apply](p7l3r3/ctrl-pair.txt), [apply/pack](p7l3r3/apply-pack.txt), [shared-row sweep](p7l3r3/router-tiles.txt), [register-row sweep](p7l3r3/reg-router.txt)

Real graph branches split the independent message half from the state half of the controller dot. Moving the fork from the mix to the preceding projection/FFN sum, then before attention, gave 3.071, 3.468 and 4.862 us of concurrent work per matched controller. The earliest fork had the most overlap and ran at 222.80 tok/s, below the 224.14 tok/s mix-local fork. Another schedule batched message halves from a complete pass into a 64-by-48 grid, twice per token. The side stream ran about 19.51 us per pass, with 4.21 us of measured overlap. Both exact pass schedules ran near 221 tok/s. Combining those prepared terms with the mixed-state down dot changed the logits with graphs both enabled and disabled. The default excludes that variant. [Branch screens](p7l3r3/branches.txt), [measured overlap](p7l3r3/controller-trace.txt), [pass timings](p7l3r3/layer-and-pass.txt), [pass screen](p7l3r3/pass-screen.txt)

A further 20 projection/attention schedules let the last projection tile for each KV head continue directly into attention. The fastest took 26.059 us against about 23.24 us for the two graph kernels. The exact FFN state-update-to-residual-mix experiment took 6.426-8.324 us against 6.064 us for the same 46 graph pairs, depending on the residence and tile schedule. These stages and the cross-pass branches remain reproducible at their experiment tags. [Dense continuation sweep](p7l3r3/drain.txt), [control](p7l3r3/control-edges.txt), [FFN sum/mix](p7l3r3/tail-mix.txt), [matched site timings](p7l3r3/matched-edges.txt)

### Reproducing the Round 3 measurements

The archived baseline is the Release sm_86 build of 16264c6 described in Round 2. Round 3 benchmarks launch that executable with its own shared libraries. The test driver selects the 3090 before starting a CUDA process. Its four-depth run and its off/on/on/off comparisons start at or below 63 C. Each per-depth mean in the paired comparison pools two runs with three tg128 repetitions each. Quality, DSpark and operator commands use the same Q2 model and corpus. Full Nsight databases are under `build/p7l3/r3-*.sqlite`. [Test driver](p7l3r3/validate.sh), [reference build instructions](#round-2-on-gpu0), [logit driver](p7l3/logits.cpp), [DSpark trace checker](p6/mr_check.cpp), [CUDA operator fixtures](../tests/test-backend-ops.cpp)

```bash
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0
export LD_LIBRARY_PATH="$PWD/build/bin"
mkdir -p build/p7l3r3
"$HOME/opt/cuda-12.8/bin/nvcc" -O3 -std=c++17 -arch=sm_86 -rdc=true perf/p7l3r3/dependency.cu -o build/p7l3r3/dependency
build/p7l3r3/dependency
for stage in bench quality mr ops guards; do perf/p7l3r3/validate.sh "$stage" || break; done
```

At `p7l3r3-complete-layer`, set `IK_ROUTE_VISIT=3 IK_COMPLETE_LAYER=1` to exercise the 19 eligible layers. At `p7l3r3-full-ffn`, use `IK_ROUTE_VISIT=3 IK_FULL_FFN=1` for the 42 FFN visits. At `p7l3r3-resident-route`, set only `IK_ROUTE_VISIT=3`. Values 1-3 request resident blocks per SM, subject to the reported occupancy and whole-expert-grid rounding. Use `IK_ROUTER_TILE=-4` at `p7l3r3-register-router`, `IK_CTRL_BRANCH=-1` or `IK_CTRL_PAIR=1` at `p7l3r3-ctrl-pair`, and `IK_APPLY_PACK=1` at `p7l3r3-apply-pack`. The earlier `p7l3r3-mix-down` prototype predates the shared-memory alignment fix. [Route sweep](p7l3r3/route-visit.txt), [layer timing](p7l3r3/layer-and-pass.txt), [controller measurements](p7l3r3/controller-trace.txt)

At `p7l3r3-controller-branches`, `IK_CTRL_BRANCH=1, 2, 3` select progressively earlier forks (set one integer per run). At `p7l3r3-message-pass`, use `IK_MESSAGE_PASS=1` for serial, `2` for the side stream. At `p7l3r3-projection-drain`, `IK_DENSE_DRAIN=-2 IK_DRAIN_CTAS=16` selects the best dense continuation. At `p7l3r3-tail-mix`, use `IK_TAIL_MIX=-4`. The Nsight commands profile CUDA graph nodes and export SQLite for matched-site analysis. [Branch overlap](p7l3r3/controller_trace.py), [layer spans](p7l3r3/layer_spans.py), [matched sites](p7l3r3/matched_edges.py), [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

```bash
"$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys" profile -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --export=sqlite -o build/p7l3r3/repro build/bin/llama-bench -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0 -r 1
python3 perf/p7l3r3/edges.py build/p7l3r3/repro.sqlite
```
