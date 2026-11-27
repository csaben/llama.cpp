# Phase 11 Stream B

**The staged build is p9, source `72fd097e`.** The nine files in `~/inkling-live-bin-p11b-72fd097` match the frozen release byte for byte. The loader resolved all eight project libraries inside the snapshot. The manual script selects GPU0, the uncensored 2-bit target, its folded Q8_0 shortlist-32K drafter, the Q8 vision projector, 131,072 tokens, alias `inkling-mini` and port 8031. Its fallback is the p9 uncensored 128K script. The deployment passed `bash -n` and was left unexecuted. [SHA-256 manifest](snapshot.sha256), [loader and hashes](snapshot-checks.txt), [staged CUDA checks](baseline/staged-ops/), [deployment](../../live/deploy-p11b.sh).

## Before and after

Both columns refer to the same binary and release flags on the unlocked card. The figures below come from the ordinary GPU-resident p9 baseline. No independent post-staging throughput run or release speedup is claimed.

| Workload | Frozen p9 on GPU0 | Staged B |
|---|---:|---|
| Target-only tg128 at 0 / 4K / 16K / 32K / 64K | 228.09 / 215.43 / 202.96 / 185.44 / 155.05 tok/s | Identical binaries |
| Prefill at 4K / 16K / 64K | 3,322.53 / 3,053.54 / 2,325.67 tok/s | Identical binaries |
| Short plain, 3,000 output tokens on code / edit / prose / agent | 213.2 / 210.9 / 209.7 / 195.2 tok/s | Identical binaries |
| Short DSpark, same four inputs and budget | 266.2 / 287.2 / 318.3 / 319.1 tok/s | Identical binaries |
| 127,872 input + 3,000 output, plain / DSpark | 92.09 / 206.79 tok/s | Identical binaries |
| Quality on the uncensored GGUF, prefill / default decode | PPL 3.0571 / 4.1915 | Identical binaries |
| Maximum tested working slot with vision and 3,000 generated tokens | 229,376 tokens | Identical binaries |
| VRAM at deployed 128K, DSpark | GPU0 18,054 MiB process ready, 19,397 MiB peak card use. GPU1 zero B allocation | Identical binaries |

Each synthetic cell is a three-sample llama-bench measurement. The server measured generation on actual prompts. All four short DSpark answers and all three long answers matched the complete corresponding plain message and every generated token ID. The first IQ4_NL-dense run finished at the 02:02 service transition. The ordinary-allocation replay is tracked separately. [Bench data](baseline/), [short answers and IDs](baseline/short/), [long answers and IDs](long/p9-captures.tar.gz), [IQ4 measurements](iq4-p9/).

| Slot | Prompt + generated | Plain / DSpark tok/s | GPU0 DSpark ready process / peak card / min free, MiB |
|---:|---:|---:|---:|
| 131,072 | 127,872 + 3,000 | 92.09 / 206.79 | 18,054 / 19,397 / 4,773 |
| 196,608 | 193,408 + 3,000 | 78.51 / 174.42 | 21,190 / 22,515 / 1,655 |
| 229,376 | 226,176 + 3,000 | 67.42 / 147.26 | 22,758 / 24,085 / 85 |
| 245,760 | Server startup | 329.86 MiB allocation failed | No ready server |

Peak VRAM was sampled device-wide every 200 ms and includes other users of the card. The maximum tested slot is an observed lower bound on capacity at the time. [Memory and generation logs](long/).

## Verification and GPU availability

On the uncensored GGUF, frozen p9 matched single-token FP32 logits on all 2,714 checked rows. These include a 1,699-row recorded rollback trace, three 73-row boundary cases, a 40-row legacy case and six 126-row randomized cases. The native release matrix completed 36 drafted cells against their plain controls in the initial capture. All 108 replies and 41,472 IDs matched on repeated 384-token runs. The original p9 release has a complete 194-cell matrix on the historical GGUF. [Rollback checks](baseline/checks/rows.txt), [uncensored matrix audit](baseline/initial-matrix.tsv), [replayable messages and IDs](baseline/initial-matrix.tar.gz), [historical p9 matrix](../../p9/matrix/audit.txt).

GPU0 is an RTX 3090 at PCI 01:00.0 with driver 615.71.09 and a 370 W power limit. At 02:02 EST a separate server began using 21,354 MiB of GPU0. A locked check found 1,669 MiB free, below the 9,131 MiB target allocation. The orchestrator stopped the separate service and the locked GPU0 check recorded 23,030 MiB free at 03:30:22 EST. Ordinary full-model qualification resumed, along with fresh repetitions of the operator measurements made during the overlap. GPU0 measurements acquired the shared lock individually with PCI bus device ordering and visibility 0. GPU1 remained available to Stream A. [Hardware](baseline/gpu.csv), [occupancy and allocation failure](baseline/external-occupancy.txt).

On ordinary GPU allocations after the service stopped, the candidate matched single-token decode on all 1,699 rows of the recorded rollback trace, with maximum absolute logit difference zero. The stored trace supplies historical tokens. On the uncensored weights, 48 recorded next tokens differed from the greedy argmax on both frozen p9 and the candidate. Fresh served comparisons use the uncensored model's own plain answers. The candidate's uncensored default decode PPL was 4.1915. [Both native trace controls](narrow/trace-control.txt), [candidate decode PPL](narrow/decode-default.txt).

The complete 384-token old/new/new/old server crossover measured 240.9 to 248.0 tok/s on code (+2.95%), 337.4 to 348.9 on edit (+3.41%), 222.9 to 223.2 on prose (+0.13%) and 273.55 to 283.55 on agent (+3.66%). Each leg used the median of three requests per prompt. Both candidate legs exceeded both p9 legs on code, edit and agent. Every complete message and every generated ID matched the candidate and frozen plain controls. [Per-leg rates and output digests](narrow/paired.tsv), [raw paired captures](narrow/paired-captures.tar.gz).

The first native 128K candidate server generated 3,000 IDs after a 127,872-token prompt. Its full message and every ID matched frozen p9 plain. At zero depth the candidate also matched every one of the 200,058 finite FP32 logits in 385 synthetic output rows, including warmup and the three tg128 repetitions. The model head and tokenizer tables were hashed directly on the uncensored, experimental and IQ4_NL-dense GGUFs. All three share the same output-head bytes and token, token-type and merge table hashes. [Long captures and VRAM samples](long/narrow-native-captures.tar.gz), [synthetic output proof](narrow/bench-logits/tg0.txt), [model identities](baseline/model-heads.tsv).

The historical GGUF reproduces the brief's PPL references of 3.0530 prefill, 4.1979 default decode and 4.2038 MR decode. The uncensored baseline above used the same corpus and chunk settings. [Historical controls](baseline/historical-model/), [uncensored prefill](baseline/prefill-default.txt), [uncensored decode](baseline/decode-default.txt).

## Profiles and tuning

At 4K, Q2_K gate/up and Q2_0 down expert GEMMs took 28.7% and 13.2% of kernel time, and direct Q8 Flash Attention took 9.8%. At 16K these shares were 26.4%, 12.2% and 15.4%. The short edit profile's median target-graph duration at widths 1 through 7 was 4.62 / 6.92 / 8.55 / 10.36 / 12.45 / 14.59 / 16.28 ms. Widths 5 through 7 each occurred twice. Across the six width-5/6/7 captures, the Q2_K gate/up kernel took 27.8% to 28.1% of target graph time. These six captures accounted for 23.2% of the GPU time in all 42 identified target graphs. A gain confined to that kernel affects a much smaller fraction of the served request. [Per-width GPU time](p9-graph-kernel-shares.tsv).

Both cards were already unlocked when testing started. The host's archived GPU0 GEMM improved from 73.9 to 119.5 TFLOPS in BF16 and from 72.4 to 115.5 in FP16. The installed unlock project identifies floating-point tensor math with FP32 accumulation as the affected path. Disassembly of the actual p9 kernels shows integer IMMA in both expert prefill projections, integer DP4A in the default Q2_K verifier, and floating-point HMMA in the packed attention scan. The width specialization trims the static DP4A count from 320 at width eight to 200, 240 and 280 at widths five, six and seven. Instruction counts describe the binary, independent of wall time. [Host GEMM results](baseline/archived-gemm.tsv), [p9 instructions](screen/p9-sm86-instructions.tsv), [candidate instructions](screen/narrow-sm86-instructions.tsv), [installed unlock documentation](https://github.com/Zagitta/gpu-unlock/blob/7eb51bc/README.md). Phase 10 used a different card, power limit and target. [Portable kernel records for all three profiles](profiles/), [4K kernels](p9-prefill-kernels.csv), [16K kernels](p9-16k-prefill-kernels.csv), [served profile](p9-verify-kernels.csv), [target graph counts and costs](p9-graph-costs.tsv), [Phase 10 conditions](../../PHASE10_NUMBERS.md).

The first operator screens overlapped the separate server reservation. All 473 expert measurements and 291 attention measurements from that period are archived with timestamps and SHA-256 hashes. Their timings are excluded from the native performance comparisons. The repeat scripts use the same operation, routing data, kernel configuration and input sizes. [Original measurements and index](screen/overlap-operator-samples.tar.gz), [reproduction recipes](screen/native-recipes.tar.gz).

| Experiment | Outcome |
|---|---|
| Expert prefill tiles and Stream-K | Uniform routing favored a 40-column down-projection tile. With real expert IDs and activations the benefit varied by prompt, and the corpus run slowed. The production candidate removes this tile. Several forced Stream-K splits changed FP32 output bytes. |
| Verification widths 2 through 7 | Only widths 5, 6 and 7 use the new Q2_K specialization. The resulting CUDA code reduces the static instruction count at those widths. The four short-prompt native crossovers above include the kernel and policy changes together. |
| Attention scan | On frozen p9, changing only `IK_WINDOW_COLS` from 1 to 2 gave 221.80 to 250.35 tok/s at 32K (+12.87%) and 256.80 to 254.70 at 16K (-0.82%) for a 384-token edit. All four crossover legs matched the frozen plain messages and IDs. Alternative scan splits changed output bytes in some cases. [Paired native rates](screen/native-packed-deep384.tsv), [raw messages, IDs and controls](screen/native-packed-deep384.tar.gz). |
| DSpark cost and cap | The candidate scales the excess verify-cost prior by 0.70. A separate cap-seven prototype gained 0.52% to 1.20% on the four short prompts in a complete crossover before the unrelated server started. Its repeated messages and IDs matched. [Cap-seven crossover](screen/history-cap7.tsv), [replies](screen/served-screens.tar.gz). |
| Flash Attention, relative bias and microbatch | The packed Q8 conversion lacked a consistent 16K prefill gain. Relative-bias blocks of 1,024 threads exceeded launch resources. Reducing the microbatch to 256 or 384 slowed 4K prefill in the screening runs. [Flash Attention](screen/fa/), [bias](screen/bias/), [microbatch](screen/ubatch/). |

The initial backend and sanitizer results are retained with the screening evidence. Ordinary-allocation operator reruns use the same model-shaped cases and verifier widths. [Screening operator checks](screen/ops/), [eleven archived allocation failures](screen/recovery/allocation-failures.tsv).

## Reproduction and sources

The build matches 193 reference CMake options and compiler paths. From the worktree root, run `python3 perf/p11/b/configure.py` and build the targets `llama-server llama-bench llama-perplexity test-backend-ops`. Prepare the frozen p9 frontends with `P9_TOOL_DIR=/path/to/p9/tools bash perf/p11/b/prepare.sh`. The original p9 matrix entry point is `perf/p9/verify.sh matrix`; its wrapper pins GPU1 and the historical GGUF. The B runner retains its suites, budgets, repetitions, probe and summarizer and uses the requested GPU, local test port and uncensored target. [CMake check](baseline/build-options.txt), [p9 frontend hashes](baseline/frontend.sha256), [original runner](../../p9/verify.sh), [Stream B runner](matrix.sh), [ID auditor](audit.py).

Hardware change descriptions and host GEMM measurements come from the machine brief. The installed GPU-unlock revision sets GA102 tensor speed selectors. The P2P driver is relevant to the machine configuration and to Stream A. B uses one card. [Machine brief](../../PHASE11_BRIEF.md), [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock/tree/7eb51bc), [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules/tree/0d96f83).
