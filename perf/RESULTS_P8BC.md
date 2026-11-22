# Phase 8 B/C

## Controls

Physical GPU 0 (RTX 3090, 01:00.0), release Q2_K/Q2_0 target, folded 32K DSpark shortlist, Q8_0 KV, 64K slot, backend greedy sampling, cache off, EOS suppressed. Short rates are medians of three uncached requests unless indicated. Full sorted message JSON and little-endian int32 emitted IDs are checked on every repeat. The agent replay includes an assistant read call, a numbered tool result and a user follow-up asking for the edited file. [Runner](p8bc/probe.sh), [flags](p8bc/env.sh), [request and capture](spec_probe.py), [comparator](p6/r4_cmp.py).

Frozen baseline uses the Phase 7c server and all its matching libraries from `~/inkling-live-bin-p7c-401460d`. It already proposes history after a 12 or 16 token match. Historical GPU 1 rates are in the [Phase 7c matrix](RESULTS.md#phase-7c). The local build uses the SM86 CUDA 12.8 Release flags documented in [RESULTS.md](RESULTS.md).

## Sources

- Prompt lookup copies candidates from an earlier matching token sequence. [Saxena, Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding), [llama.cpp n-gram implementation](../common/ngram-map.cpp).
- Basalt compares estimated committed tokens per unit time across sources and tracks conditional success by match length. [MIT draft policy](https://github.com/jesdga95/basalt/blob/ae82a4841c95f6a05ca8ca2331ef6af7ca844aef/src/spec/draft_policy.cpp).
- Ingine groups routed rows and decodes its packed weights into BF16 MMA fragments. Its data layout and precision differ from this branch's GGUF/Q8 activation MMQ. [Apache-2.0 expert kernel](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/306eaa8a90c9840f042a59d565f6dc4536bbb5ea/native/src/kernels/prefill/wgemm.cu).

## Frozen GPU 0 baseline

Decode tok/s, 384 / 3,000 emitted tokens. All six drafted messages and ID streams matched plain at both lengths. The five standard plain prompts also reproduced the Phase 7c references across GPUs.

| Prompt | Plain | Release cap 3 |
|---|---:|---:|
| Code | 208.5 / 191.7 | 241.9 / 262.6 |
| Edit | 198.7 / 191.1 | 327.7 / 293.1 |
| Prose | 208.1 / 196.5 | 209.1 / 283.6 |
| Edit at 16K | 180.3 / 178.0 | 278.8 / 284.9 |
| Edit at 32K | 166.2 / 164.9 | 243.5 / 286.7 |
| Agent | 191.8 / 192.1 | 280.0 / 308.8 |

[384-token captures](p8bc/baseline/384/), [3,000-token captures](p8bc/baseline/3000/), [GPU and driver](p8bc/baseline/device.csv).

## Draft selection trials

The three-token minimum tried too many weak matches on short prose. With a five-token minimum, the policy estimates continuation success in three suffix-length bands. A rejection contributes one failed trial because the target never evaluates the remaining suffix against the corrected history. Width selection uses estimated committed tokens per measured millisecond, including plain/model alternatives and a limited probe of a longer copy. A shorter suffix can corroborate a paid DSpark prediction. On the long prose trace, the model's margin between 2 and 3 yielded 85/136 correct without history agreement and 14/15 with agreement. The rule raises the estimated confidence only along the agreeing prefix and leaves candidate IDs and target sampling intact. Counts use emitted history for offline analysis. [Margin counts](p8bc/trials/model-confidence.txt), [three-token trial](p8bc/trials/three-token-min.txt), [selection code](../common/speculative.cpp).

| 384 tokens, code / edit / prose / agent | tok/s |
|---|---|
| Three-token minimum, cap 3 | 242.3 / 334.0 / 205.3 / 280.2 |
| Five-token minimum with measured costs, cap 3 | 243.7 / 337.2 / 211.2 / 288.2 |
| Five-token minimum with measured costs, cap 6 | 242.7 / 325.9 / 211.5 / 290.3 |
| Cap 6 with five-to-seven-row IQ4 projections | 244.6 / 335.0 / 211.2 / 288.8 |
| Cap 6 plus DSpark/history agreement, five repeats | 244.0 / 338.6 / 214.1 / 290.5 |

The final trial measured 263.4 / 293.3 / 288.7 / 314.6 tok/s at 3,000 tokens. All eight messages and their IDs matched frozen plain. Each variant uses its own live run, so small differences include timing variation. [Trial captures](p8bc/trials/), [exact verifier costs](p8bc/windows/).

## Stream B verifier checks

Both the release and default environments passed all 2,714 rollback/boundary rows, six CLI greedy probes and 1,108 CUDA tests. The historical failing trace compares the kept logits bit for bit. As in the Phase 7c control, two tokens recorded by the old failing server are not today's single-row argmax. Prefill PPL was 3.0530 in both runs and decode PPL was 4.1979 (plain) / 4.2038 (multi-row flags). [Release](p8bc/checks/release-rows.txt), [defaults](p8bc/checks/default-rows.txt), [PPL and plain decode rates at four depths](p8bc/checks/).

Eight mixed-width runs at prefixes 200 / 4,096 / 16,384 / 32,768 with packing on/off and randomized rollback each matched all 182 kept rows (1,456 in total). The rebuilt and frozen libraries produced identical FP32 bytes for all 88 full-vocabulary depth rows and 128 interleaved sequence rows. Both synchronization checks passed. CUDA passed 4,229 expanded cases across matrix multiplication, router/attention/controller operations, SITU, CONCAT and relative bias. [Depths](p8bc/checks/depths.txt), [hashes](p8bc/checks/logits.txt), [sanitizer and operator totals](p8bc/checks/sync-ops.txt).

Run `CHECKS_GPU=0 CHECKS_OUT=build/p8-check bash perf/p6/r4_checks.sh` for the baseline battery. The width stress uses `mr_check`, two consecutive copies of `perf/ppl.txt`, `MR_RS_A=0 MR_RS=6 MR_SIZES=2,3,4,5,6,7 MR_RAND=37 MR_PLAIN_PCT=0 MR_CTX=34816`, 64 windows, and both `IK_WINDOW_COLS` settings. The frozen comparisons use the [full-logit driver](p7l3/logits.cpp).

## Complete Phase 7c request matrices on GPU 0

All 360 drafted requests (347,520 emitted token IDs) matched the full canonical message and the ID bytes from their same-flags plain server. The 384-token suite contributed 280 requests, including repeats, and the 3,000-token suite contributed 80. The main and fixed suites include the agent replay at both budgets. The TSV has 194 cells including plain controls, with rate, acceptance counts, and independently checkable hashes. [All cells](p8bc/matrix/cells.tsv), [raw rates and repeats](p8bc/matrix/), [runner](p8bc/matrix.sh), [byte comparator and totals](p8bc/summarize.py).

| Cap 6, adaptive | Plain 384 | Hybrid 384 | Plain 3,000 | Hybrid 3,000 |
|---|---:|---:|---:|---:|
| code | 205.9 | 242.7 | 195.3 | 263.0 |
| edit | 194.8 | 333.1 | 189.7 | 294.0 |
| prose | 204.0 | 213.0 | 179.1 | 289.9 |
| edit@16K | 178.5 | 274.5 | 162.5 | 267.6 |
| edit@32K | 162.1 | 203.9 | 152.5 | 242.4 |
| agent | 186.8 | 280.2 | 174.0 | 270.9 |

Eight of the 20 guarded short additional-prose medians fell below their matrix plain controls (worst was review, -4.9%). In the subsequent six-process prose crossover, every cap-6 median exceeded both surrounding plain medians. Averaged over the two processes, cap 6 was 0.9% faster than frozen cap 3 on science and 0.5-1.2% slower on the other four. GPU 0 reported software thermal throttling during the matrix. [Paired prose](p8bc/paired/prose.txt), [process-start clock samples](p8bc/paired/gpu.csv).

## Frozen release crossover

Same six prompts in the same order. Each length ran frozen cap 3, new cap 6, new cap 6, frozen cap 3. Each short rate is a five-request process median, each long rate a single request. Percentages compare the means of the two process results per implementation. Control ranges matter, especially after the 32K short prefill. [384 tokens](p8bc/paired/384.txt), [3,000 tokens](p8bc/paired/3000.txt).

| Prompt | Frozen 384 range | Cap 6 range | Change | Frozen 3,000 range | Cap 6 range | Change |
|---|---:|---:|---:|---:|---:|---:|
| code | 244.4-248.0 | 244.1-245.3 | -0.6% | 268.5-269.7 | 267.6-268.0 | -0.5% |
| edit | 332.4-336.1 | 337.2-338.7 | +1.1% | 296.1-297.0 | 297.6-298.7 | +0.5% |
| prose | 211.2-214.3 | 212.9-213.9 | +0.3% | 287.6-288.4 | 294.1-294.4 | +2.2% |
| edit@16K | 239.6-284.5 | 261.9-278.2 | +3.1% | 289.6-290.5 | 290.3-290.5 | +0.1% |
| edit@32K | 216.0-250.2 | 213.9-238.7 | -2.9% | 292.7-292.7 | 293.9-294.3 | +0.5% |
| agent | 268.1-287.2 | 263.3-294.0 | +0.4% | 319.8-320.0 | 319.6-323.1 | +0.5% |

The isolated 384-token agent replay averaged 285.5 tok/s frozen, 290.6 new cap 3 and 294.5 new cap 6. The process order was frozen/3/6/6/3/frozen. [Isolated agent](p8bc/paired/agent-384.txt).

The isolated 3000-token agent replay averaged 320.1 tok/s frozen, 319.9 new cap 3 and 319.7 new cap 6. The process order was frozen/3/6/6/3/frozen. [Isolated agent](p8bc/paired/agent-3000.txt).

A separate tool-enabled replay stopped naturally at 777 tokens. Cap 6 and plain produced identical full messages and IDs at 241.0 and 190.6 tok/s respectively. [Natural tool reply](p8bc/paired/natural.txt).

## Stream C baseline

The saved Stream B runtime ran three llama-bench repetitions per length at batch 2,048, microbatch 512, flash attention and Q8_0 KV. At 4K / 16K / 32K it measured 2,892 / 2,675 / 2,443 prompt tok/s. Three uncached tool-follow-up requests at each length had median client TTFT of 1.755 / 6.820 / 14.636 seconds for 4,171 / 16,459 / 32,843 actual prompt tokens. The 8 x 2,048 WikiText control reproduced 3.0530 PPL. The tool replay includes tokenization, prompt evaluation and streaming to the first visible or reasoning delta. [Benchmark samples, client timings and PPL](p8bc/prefill/base/), [benchmark runner](p8bc/prefill.sh), [tool replay](p8bc/ttft.py).

Nsight Systems with CUDA graph node tracing attributed 58.3% of 4K GPU kernel time to MMQ, 8.7% to FlashAttention and 6.7% to relative bias. At 32K those shares were 49.6%, 19.7% and 7.8%. At 4K the 32-column Q2_K and Q2_0 expert kernels alone consumed 480 and 164 ms out of 1,380 ms. [4K profile](p8bc/prefill/base/pp4096_cuda_gpu_kern_sum.csv), [16K profile](p8bc/prefill/base/pp16384_cuda_gpu_kern_sum.csv), [32K profile](p8bc/prefill/base/pp32768_cuda_gpu_kern_sum.csv), [profiler wrapper](p6/prof.sh).

## Stream C kernel trials

The width sweep used the same 4K prefill and microbatch 512. The saved B control processes averaged 2,858 and 2,892 tok/s. Unpacking Q2_K in the MMA fragment, with the original shared-memory spacing, averaged 2,927 and 2,947. Expert widths 16 / 24 / 48 / 64 measured 2,733 / 2,885 / 2,940 / 2,901. Dense IQ4 widths 32 / 64 / 96 measured 2,826 / 2,930 / 2,941. Halving the FlashAttention query tile measured 2,798. The existing dense and attention tile choices were retained. Absolute rates here are sensitive to process order. [Per-run trials](p8bc/prefill/trials.txt), [MMQ dispatch](../ggml/src/ggml-cuda/mmq.cu), [FlashAttention dispatch](../ggml/src/ggml-cuda/fattn.cu).

Computing the relative-bias lookup tables four heads at a time measured 3,073 / 3,051 versus 2,962 between those trials. Packing each Q2_K row into 36 shared words gave 3,100 / 3,083. With compact rows, whole expert tiles measured 3,283. Scheduling row tiles of an expert together gave 3,300 versus 3,256 / 3,253 in the surrounding column-first runs. Widening those tiles to 48 reduced both rates. A separate host relative-position construction trial measured 2,962 / 2,969 at 16K against 2,977 / 2,977 with the existing implementation and was discarded. [Trials](p8bc/prefill/trials.txt).

The integer Q2_K path keeps 16 packed words and 16 scale/min pairs per row, then unpacks four two-bit groups from one matrix fragment load. The dot-product operands and the order of floating-point accumulation are preserved. This is an independent implementation in llama.cpp's integer MMQ. The existing direct scheduler handles full-K expert tiles when Stream-K would already assign one block per tile. The four-head bias kernel reuses each table row, position and mask load and keeps the existing dot-product order. [Q2_K loader](../ggml/src/ggml-cuda/mmq-load-tiles.cuh), [fragment arithmetic](../ggml/src/ggml-cuda/mmq-vec-dot.cuh), [tile scheduler](../ggml/src/ggml-cuda/mmq.cuh), [relative bias](../ggml/src/ggml-cuda/inkling.cu), [Ingine reference](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/306eaa8a90c9840f042a59d565f6dc4536bbb5ea/native/src/kernels/prefill/wgemm.cu).

## Stream C quality and profiles

The combined 512-microbatch build at `p8c-bias` reproduced prefill PPL 3.0530 +/- 0.06532 on the eight WikiText chunks. Its first three-run throughput means were 3,302 / 3,017 / 2,720 prompt tok/s. At 4K, Nsight recorded 1,199 ms of GPU kernel work against 1,380 ms at baseline, with 19,840 launches in both. The four-head kernel took 50 versus 92 ms. [PPL and final Nsight captures](p8bc/prefill/final/), [per-category times at all lengths](p8bc/prefill/kernels.tsv).

## Stream C release crossover

GPU 0 ran B / C / C / B at the same batch 2,048 and microbatch 512. Each process ran three llama-bench samples and a fresh 64K server with three uncached tool requests per length. Throughput is the mean of process means. TTFT is the mean of process medians. The source files and chat template gave the same 4,171 / 16,459 / 32,843 actual tokens in every process. [All samples](p8bc/prefill/paired.tsv), [replay](p8bc/ttft.py).

| Prompt | B prompt tok/s | C prompt tok/s | Change | B client TTFT | C client TTFT | Change |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 2,906 | 3,298 | +13.5% | 1.757 s | 1.541 s | -12.3% |
| 16K | 2,683 | 3,015 | +12.4% | 6.827 s | 6.008 s | -12.0% |
| 32K | 2,437 | 2,719 | +11.6% | 14.630 s | 13.071 s | -10.7% |

The 16K profile saved 562 ms in Q2_K experts, 30 ms in Q2_0 experts and 141 ms in relative bias. Dense IQ4 MMQ and FlashAttention were within 2% of the original kernel times. [Nsight category totals](p8bc/prefill/kernels.tsv).

The final CUDA build passed 3,779 matrix and routed-matrix cases plus 13 relative-bias cases. These include dedicated ragged 257-row routed matrices and a 129-token, 1,025-key bias test. All 216 captured full-vocabulary FP32 rows matched B byte for byte, including interleaved sequences and prefixes through 32K. In a further 728 kept target rows at four depths, mixed-width decode reproduced single-token decode with maximum logit difference zero. [Operation and window results](p8bc/prefill/checks.txt), [FP32 capture hashes](p8bc/prefill/logits.tsv), [capture driver](p7l3/logits.cpp), [width and rollback driver](p6/mr_check.cpp).
