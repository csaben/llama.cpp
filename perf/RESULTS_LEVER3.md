# Phase 7 GPU0 plain decode

[Round 2 results and streaming-floor measurements](#round-2-on-gpu0)

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

### Streaming floor for one real route

One resident kernel evaluated the **193 quantized matvecs in 2.692 ms**. A CUDA graph with the 193 production matvec launches took **2.871 ms** before it and **2.892 ms** after it. The measured difference was **0.179-0.200 ms/token**. The 300 tok/s target allows 3.333 ms for everything, leaving 0.642 ms after the resident dot workload for unhidden attention, routing, controllers, recurrent state and other work. [Raw event batches](p7l3r2/floor-first.txt), [launch sweep](p7l3r2/floor-sweep.txt), [brief](PHASE7_LEVER3_BRIEF.md#round-2-after-tag-p7l3-final-219tok)

| Schedule on the captured weights | Median ms/token | Logical weight GB/s |
|---|---:|---:|
| Single resident kernel, real quantized dots | 2.692 | 745.5 |
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
| Entire dense chain and direct router in one resident launch | At 24 visits, QKV, attention, WO, state update and top-8 router selection took 49.234 us with shared-weight staging or 49.621 us with L2 prefetch, against 40.797 us for the winning single dense kernel followed by the native router. At the other 24 visits the controller intervenes. Both variants matched the full reference logits. | `p7l3r2-whole-visit-router` |
| Extend upstream through residual mix | A cooperative launch overlapped input-weight staging with mix, normalization and Q8 packing. It ran at 24 visits/token. Preloading up to 56 rows/worker cost throughput. | `p7l3r2-visit-extensions` |
| Longer cache | Separate graph streams for the gate and scan lost throughput. A 384-thread block running eight scan warps and four gate warps was faster than eight plus eight in the screen. | `p7l3r2-visit-extensions` |
| Lossless dense packing | The loader joined each layer's QKV and WO allocations and moved the same IQ4_NL scales to the end of each row. It needed 363,377,664 extra bytes. The small throughput change did not justify keeping the second layout. | `p7l3r2-dense-layout` |
| Lossless expert packing and output tiles | Packing Q2_0 codes for two fewer byte permutes per eight values took 12.397 us for down plus sum/conv against 12.168 us on native weights. A native four-row, eight-slot fused tile took 12.701 us at the 44 Q2_0 visits. | `p7l3r2-expert-layout` |

The measurements above use live router outputs, KV caches and recurrent state unless explicitly labeled as the streaming-floor experiment. Nsight intervals run from the first kernel start through the last kernel end at each visit over 119 complete token intervals. A separate full-logit run checked the dense resident and staged prototypes. The early expert screen with no eligible fusions was excluded after inspecting its trace. [First visit](p7l3r2/visit-first.txt), [queue sweep](p7l3r2/queue-screen.txt), [late-gate traces](p7l3r2/gate-traces.txt), [gate geometry](p7l3r2/stage-screen.txt), [dense logit hashes](p7l3r2/dense-hashes.txt), [router](p7l3r2/router-trace.txt), [mix](p7l3r2/mix-screen.txt), [long-cache streams](p7l3r2/long-screen.txt), [warp counts](p7l3r2/warps.txt), [dense packing](p7l3r2/pack-screen.txt), [expert screen](p7l3r2/expert-layout-screen.txt), [expert and mix coverage](p7l3r2/expert-trace.txt)

The expert epilogue initially rounded the final scale and residual add differently from the production FFN. Restoring its single fused multiply-add gave the baseline logit hash for native and packed expert tiles, and for the standalone packed matvec. The corrected native/packed full-tail timings were 12.671/12.354 us. Duplicating the 22 Q2_0 physical expert tensors would cost 2,102,132,736 bytes. Both experimental duplicates were discarded from the selected implementation. [Corrected expert traces](p7l3r2/expert-exact-trace.txt), [resident router trace](p7l3r2/resident-router-trace.txt), [resident router benchmark](p7l3r2/resident-router-screen.txt), [coverage analysis](p7l3r2/extended_trace.py)

The single-launch prototypes use a cooperative resident grid, per-task release/acquire counters, and the existing Inkling scan and quantized arithmetic. The instruction-stream design drew on Hazy Research's Llama-3.2-1B work and the Mirage Persistent Kernel paper. Mirage calls its representation an SM-level graph. Neither external implementation was linked. The experiments used local GA102 measurements for every promotion decision. [Hazy Research design](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), [Cheng et al., MPK](https://arxiv.org/abs/2512.22219), [CUDA asynchronous copies](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async), [Inkling layer graph](../src/models/inkling2.cpp), [attention](../ggml/src/ggml-cuda/inkling.cu), [projection and shortconv](../ggml/src/ggml-cuda/inkling-fmv.cu)
