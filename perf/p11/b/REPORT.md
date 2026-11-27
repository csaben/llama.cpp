# Phase 11 Stream B

**The staged build is p9, source `72fd097e`.** The nine files in `~/inkling-live-bin-p11b-72fd097` match the frozen release byte for byte. The loader resolved all eight project libraries inside the snapshot. The manual script selects GPU0, the uncensored 2-bit target, its folded Q8_0 shortlist-32K drafter, the Q8 vision projector, 131,072 tokens, alias `inkling-mini` and port 8031. Its fallback is the p9 uncensored 128K script. The deployment script passed `bash -n` and was left unexecuted. Two fresh drafted runs of the staged vision server matched frozen p9 and the staged plain control on all 24 replies and 9,216 IDs. [SHA-256 manifest](snapshot.sha256), [loader and hashes](snapshot-checks.txt), [staged vision-server probes](staged-controls/p9.txt), [deployment](../../live/deploy-p11b.sh).

The experimental candidate passed the full token matrix and improved short edit throughput by 3.41%. Its paired 128K DSpark mean fell 3.00%, from 198.34 to 192.39 tok/s. At the 07:30 EST cutoff, three stock rollback cases and the release-flags PPL crossovers were still open. [Candidate matrix](matrix-narrow/audit.txt), [short crossover](narrow/paired.tsv), [128K crossover](long/narrow-native.tsv).

## Before and after

The table gives the ordinary GPU-resident p9 baseline on the unlocked card. The staged build uses those exact binaries and release flags.

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

Each synthetic cell is a three-sample llama-bench measurement. The server measured generation on actual prompts. All four short DSpark answers and all three long answers matched the complete corresponding plain message and every generated token ID. The optional IQ4_NL-dense zero-depth run finished at the 02:02 service transition. Its ordinary-allocation repeat did not finish before cutoff, so the report gives no IQ4 rate. [Bench data](baseline/), [short answers and IDs](baseline/short/), [long answers and IDs](long/p9-captures.tar.gz), [IQ4 measurements](iq4-p9/).

| Slot | Prompt + generated | Plain / DSpark tok/s | GPU0 DSpark ready process / peak card / min free, MiB |
|---:|---:|---:|---:|
| 131,072 | 127,872 + 3,000 | 92.09 / 206.79 | 18,054 / 19,397 / 4,773 |
| 196,608 | 193,408 + 3,000 | 78.51 / 174.42 | 21,190 / 22,515 / 1,655 |
| 229,376 | 226,176 + 3,000 | 67.42 / 147.26 | 22,758 / 24,085 / 85 |
| 245,760 | Server startup | 329.86 MiB allocation failed | No ready server |

Peak VRAM was sampled device-wide every 200 ms and includes other users of the card. The maximum tested slot is an observed lower bound on capacity at the time. [Memory and generation logs](long/).

## Candidate serving and numerical checks

| Verified DSpark crossover | Frozen p9 tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Code, 384 output | 240.90 | 248.00 | +2.95% |
| Edit, 384 output | 337.40 | 348.90 | +3.41% |
| Prose, 384 output | 222.90 | 223.20 | +0.13% |
| Agent, 384 output | 273.55 | 283.55 | +3.66% |
| 127,872 input + 3,000 output, vision slot | 198.34 | 192.39 | -3.00% |

The short trials used the median of three requests per prompt in each old/new/new/old leg. Both candidate legs exceeded both p9 legs on code, edit and agent. Every complete message and generated token ID matched the candidate and frozen plain controls. [Short crossover and output digests](narrow/paired.tsv), [raw paired captures](narrow/paired-captures.tar.gz), [long crossover and VRAM](long/narrow-native.tsv).

## Native qualification at the cutoff

The candidate `caec8de6` passed all 194 release-matrix cells. The 360 drafted replies contained 347,520 generated IDs. All matched both same-flags plain and the fresh frozen p9 plain control, including the complete assistant messages. The archived matrix was re-extracted and independently replayed through the p9 summarizer and ID audit. [Candidate matrix](matrix-narrow/audit.txt), [raw captures](matrix-narrow/captures.tar.gz). Frozen p9 completed 145/194 cells on the uncensored GGUF, with 271 drafted replies and 211,320 IDs matched. [Frozen p9 checkpoint](matrix-p9/progress.tsv).

The candidate completed 8/11 stock rollback and boundary cases (2,336/2,714 rows). Including extra width checks, all 3,064 tested full-vocabulary FP32 rows matched single-token decode with maximum absolute difference zero. The deployed-width sweep contributed 728 rows after 200, 4,096, 16,384 and 32,768 input tokens. [Numerical rows](narrow/checks/native-rows.tsv), [complete logs](narrow/checks/native-logs.tar.gz).

On the uncensored model with release flags, prefill completed 3/4 legs (3.0571, 3.0571, 3.0571, pending) and decode had no completed legs. Both runtimes separately measured 4.1915 with MR disabled for the default decode control. [Quality controls](quality/), [candidate default decode](narrow/decode-default.txt).

After GPU0 became available again, 160/473 expert samples and 63/291 attention samples reproduced the original output hashes. Six of the eleven configurations that originally ran out of memory completed a verified native rerun. The remaining 313 expert and 228 attention samples were still open when testing stopped. The native archive contains nine timing groups per completed case. None of the retried isolated operator crossovers finished all four legs, so their rates do not enter the release comparison. [Rerun coverage](screen/recovery/operator-coverage.tsv), [native samples](screen/native-operator-samples.tar.gz), [allocation reruns](screen/recovery/native-retries.tsv).

The target-only throughput crossovers did not finish all four timing legs. The grouped 4K, 16K and 64K prefill output checks matched 344,064 input token IDs and all 168 rows of 200,058 finite FP32 logits. [Prefill checks and output digests](narrow/bench-grid/pp-output-proof.tsv), [individual timing legs and matched partial server probes](narrow/partial-captures.tar.gz).

Near the 128K slot limit, the four DSpark legs measured 191.53 / 177.69 / 207.09 / 205.14 tok/s. The two candidate legs spanned 29.40 tok/s. [Long-run rates, IDs and VRAM](long/narrow-native.tsv).

The same 128K workload completed three plain-decode legs at 85.13 / 91.01 / 83.51 tok/s. The closing p9 leg remained open at cutoff. [Long-run rates, IDs and VRAM](long/narrow-native.tsv).


Frozen p9 passed the stock rollback/boundary battery on the uncensored target, with all 2,714 full-vocabulary rows matching single-token decode. The historical trace supplied 48 next tokens that differ from the uncensored greedy argmax on both runtimes. Fresh served comparisons use the uncensored model's own plain replies. [Frozen rows](baseline/checks/native-rows.tsv), [trace comparison](narrow/trace-control.txt).

The candidate also matched 385 full-vocabulary rows from the standalone zero-depth benchmark, covering warmup and three tg128 repetitions. The output head and token, token-type and merge tables match across the uncensored, experimental and IQ4 GGUFs. [Synthetic outputs](narrow/bench-logits/tg0.txt), [GGUF hashes](baseline/model-heads.tsv).

With the vision projector loaded, the candidate generated a 3,000-token reply after 193,408 input tokens. The full message and every ID matched frozen p9 plain. GPU0 used 21,190 MiB at server readiness and peaked at 22,635 MiB device-wide. Stream B allocated zero on GPU1. [Capacity output check](long/narrow-slot-196608-dspark.p9-equal), [memory samples](long/narrow-slot-196608-dspark.txt).

On the candidate, historical prefill and default decode reached 3.0530 and 4.1979. The historical MR-decode check remained open. [Candidate historical controls](narrow/historical-model/). Frozen p9 on the historical GGUF reproduces the brief's PPL references of 3.0530 prefill, 4.1979 default decode and 4.2038 MR decode. The uncensored baseline above used the same corpus and chunk settings. [Historical controls](baseline/historical-model/), [uncensored prefill](baseline/prefill-default.txt), [uncensored decode](baseline/decode-default.txt).

GPU0 is an RTX 3090 at PCI 01:00.0 with driver 615.71.09 and a 370 W power limit. At 02:02 EST a separate server began using 21,354 MiB of GPU0. A locked check found 1,669 MiB free, below the 9,131 MiB target allocation. The orchestrator stopped the separate service and the locked GPU0 check recorded 23,030 MiB free at 03:30:22 EST. Ordinary full-model qualification resumed, along with fresh repetitions of the operator measurements made during the overlap. After 03:30, both streams continued using the shared lock for each GPU0 measurement. Stream B pinned CUDA visibility to physical GPU0. The 07:30 stop left the remaining native reruns incomplete. GPU1 remained available to Stream A. [Hardware](baseline/gpu.csv), [occupancy and allocation failure](baseline/external-occupancy.txt).

## Profiles and tuning

At 4K, Q2_K gate/up and Q2_0 down expert GEMMs took 28.7% and 13.2% of kernel time, and direct Q8 Flash Attention took 9.8%. At 16K these shares were 26.4%, 12.2% and 15.4%. The short edit profile's median target-graph duration at widths 1 through 7 was 4.62 / 6.92 / 8.55 / 10.36 / 12.45 / 14.59 / 16.28 ms. Widths 5 through 7 each occurred twice. Across the six width-5/6/7 captures, the Q2_K gate/up kernel took 27.8% to 28.1% of target graph time. These six captures accounted for 23.2% of the GPU time in all 42 identified target graphs. A gain confined to that kernel affects a much smaller fraction of the served request. [Per-width GPU time](p9-graph-kernel-shares.tsv).

Both cards were already unlocked when testing started. The host's archived GPU0 GEMM improved from 73.9 to 119.5 TFLOPS in BF16 and from 72.4 to 115.5 in FP16. The installed unlock project identifies floating-point tensor math with FP32 accumulation as the affected path. Disassembly of the actual p9 kernels shows integer IMMA in both expert prefill projections, integer DP4A in the default Q2_K verifier, and floating-point HMMA in the packed attention scan. The width specialization trims the static DP4A count from 320 at width eight to 200, 240 and 280 at widths five, six and seven. Instruction counts describe the binary, independent of wall time. [Host GEMM results](baseline/archived-gemm.tsv), [p9 instructions](screen/p9-sm86-instructions.tsv), [candidate instructions](screen/narrow-sm86-instructions.tsv), [installed unlock documentation](https://github.com/Zagitta/gpu-unlock/blob/7eb51bc/README.md). Phase 10 used a different card, power limit and target. [Portable kernel records for all three profiles](profiles/), [4K kernels](p9-prefill-kernels.csv), [16K kernels](p9-16k-prefill-kernels.csv), [served profile](p9-verify-kernels.csv), [target graph counts and costs](p9-graph-costs.tsv), [Phase 10 conditions](../../PHASE10_NUMBERS.md).

The initial isolated expert and scan screens overlapped the separate server reservation. All 473 expert measurements and 291 attention measurements from that period are archived with timestamps and SHA-256 hashes. Their timings are excluded from the native performance comparisons. The repeat scripts use the same operation, routing data, kernel configuration and input sizes. [Original measurements and index](screen/overlap-operator-samples.tar.gz), [additional screening diagnostics](screen/overlap-additional.tar.gz), [reproduction recipes](screen/native-recipes.tar.gz).

| Experiment | Outcome |
|---|---|
| Expert prefill tiles and Stream-K | Uniform routing favored a 40-column down-projection tile. With real expert IDs and activations the benefit varied by prompt, and the corpus run slowed. The production candidate removes this tile. Several forced Stream-K splits changed FP32 output bytes. |
| Verification widths 2 through 7 | Only widths 5, 6 and 7 use the new Q2_K specialization. The resulting CUDA code reduces the static instruction count at those widths. The four short-prompt native crossovers above include the kernel and policy changes together. |
| Attention scan | On frozen p9, changing only `IK_WINDOW_COLS` from 1 to 2 gave 221.80 to 250.35 tok/s at 32K (+12.87%) and 256.80 to 254.70 at 16K (-0.82%) for a 384-token edit. All four crossover legs matched the frozen plain messages and IDs. Alternative scan splits changed output bytes in some cases. [Paired native rates](screen/native-packed-deep384.tsv), [raw messages, IDs and controls](screen/native-packed-deep384.tar.gz). |
| DSpark cost and cap | The candidate scales the excess verify-cost prior by 0.70. A separate cap-seven prototype gained 0.52% to 1.20% on the four short prompts in a complete crossover before the unrelated server started. Its repeated messages and IDs matched. [Cap-seven crossover](screen/history-cap7.tsv), [replies](screen/served-screens.tar.gz). |
| Flash Attention, relative bias and microbatch | The packed Q8 conversion lacked a consistent 16K prefill gain. Relative-bias blocks of 1,024 threads exceeded launch resources. Reducing the microbatch to 256 or 384 slowed 4K prefill in the screening runs. [Flash Attention](screen/fa/), [bias](screen/bias/), [microbatch](screen/ubatch/). |

The native candidate passed all 18 stock backend cases for the actual expert projection shapes and all eight INKLING_ATTN cases. [Model-shaped CUDA tests](narrow/ops/model-shapes.txt), [attention tests](narrow/ops/INKLING_ATTN.txt). The initial memcheck and synccheck screens reported zero errors. Their ordinary-allocation repeats were incomplete at cutoff. [Screening operator checks](screen/ops/), [eleven archived allocation failures](screen/recovery/allocation-failures.tsv).

## Reproduction and sources

All 343 public CMake cache entries match the reference in value and type after worktree-path substitution. Both use Unix Makefiles, Release, CUDA architecture 86 and the same C, C++ and CUDA compilers. All four requested targets built. From the worktree root, run `python3 perf/p11/b/configure.py` and build the targets `llama-server llama-bench llama-perplexity test-backend-ops`. Prepare the frozen p9 frontends with `P9_TOOL_DIR=/path/to/p9/tools bash perf/p11/b/prepare.sh`. The p9 entry point `perf/p9/verify.sh matrix` uses GPU1 and the historical GGUF. The B runner retains its suites, budgets, repetitions, probe and summarizer and uses the requested GPU, local test port and uncensored target. [Full cache comparison](baseline/full-cache.txt), [four completed targets](baseline/final-build.txt), [p9 frontend hashes](baseline/frontend.sha256), [original runner](../../p9/verify.sh), [Stream B runner](matrix.sh), [ID auditor](audit.py).

Hardware change descriptions and host GEMM measurements come from the machine brief. The installed GPU-unlock revision sets GA102 tensor speed selectors. The P2P driver is relevant to the machine configuration and to Stream A. B uses one card. [Machine brief](../../PHASE11_BRIEF.md), [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock/tree/7eb51bc), [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules/tree/0d96f83).
