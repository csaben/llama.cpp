# Phase 11, Stream A

**The two-GPU output head reached 245.84 tok/s in plain decode and 355.20 tok/s on the short DSpark edit.** Plain decode cleared 241 by 4.84 tok/s, a modest gain. The DSpark edit cleared 351 by 4.20 tok/s. Source `a16c61e5` was staged at 07:17 EST in `~/inkling-live-bin-p11a-a16c61e5`. [Nine-file manifest](snapshot.sha256), [clean build provenance](provenance.txt).

## Uncensored two-bit throughput

| Workload, tok/s | p9, GPU1 | Split head, GPU1 + GPU0 | Change |
|---|---:|---:|---:|
| Plain decode, zero depth | 239.93 | 245.84 | +2.46% |
| Plain decode, 4K | 220.51 | 225.51 | +2.27% |
| Plain decode, 16K | 208.03 | 212.40 | +2.10% |
| Plain decode, 32K | 191.47 | 195.41 | +2.06% |
| Plain decode, 64K | 161.83 | 164.23 | +1.48% |
| DSpark edit, 384 generated | 348.40 | 355.20 | +1.95% |
| DSpark edit, 3,000 generated | 295.60 | 301.10 | +1.86% |
| Prefill, 4K | 3,527.25 | 3,582.34 | +1.56% |
| Prefill, 16K | 3,218.00 | 3,258.62 | +1.26% |
| Prefill, 64K | 2,441.15 | 2,445.14 | +0.16% |
| Vision slot, 128,064 prompt tokens | 1,708.43 | 1,693.26 | -0.89% |
| DSpark after 128,064 tokens, 3,000 generated | 206.37 | 208.70 | +1.13% |

Each row uses old/new/new/old after the unrelated service exited. Synthetic legs have three samples per depth. The edit legs use medians of three uncached replies to a 1,025-token prompt. The near-full vision legs each make one new request. All served replies matched p9 plain on the full message and every ID. [All four legs](performance.tsv), [near-full timings](capacity-paired.tsv), [physical GPU telemetry](telemetry.tsv).

For dense IQ4_NL at 8K with 128/128 microbatches, plain decode went from 196.35 to 201.95 tok/s. The short edit went from 246.15 to 245.00 tok/s. All twelve replies matched IQ4 p9 plain. [Four-bit legs](iq4-release/performance.tsv).

## Capacity and quality

| Measure | p9 | Split head |
|---|---:|---:|
| Largest verified vision slot at 512/512 | 253,952 | 253,952 |
| Uncensored PPL, prefill / decode / MR decode | 3.0571 / 4.1915 / 4.1877 | 3.0571 / 4.1915 / 4.1877 |
| Calibrated PPL, prefill / decode / MR decode | 3.0530 / 4.1979 / 4.2038 | 3.0530 / 4.1979 / 4.2038 |
| 128K server peak, physical GPU1 | 18,278 MiB | 18,278 MiB |
| 128K server peak, physical GPU0 | 0 MiB | 416 MiB |

Both servers completed 250,944 prompt tokens and 3,000 generated tokens in the largest slot. Both matched the independent plain request and failed allocation at the next tested size, 262,144. The calibrated model supplies the brief's historical PPL figures. [Capacity evidence](capacity/), [PPL comparisons](ppl.tsv), [model differences](model-comparison.txt).

**All 194 release-matrix cells passed.** The 360 drafted replies matched same-flags plain and frozen p9 on 347,520 ordered IDs and the complete messages. The rollback/boundary battery matched 2,714 logit rows, plus 217 rows at widths 2-7 after a 16K prefix. Backend tests passed 64/64 IQ4 and 1,110/1,110 routed cases. The head test, memcheck and synccheck each matched 589,824 FP32 values at widths 1-8, with zero sanitizer errors. [Matrix audit](release-matrix/audit.txt), [tested placements](release-matrix/placement.tsv), [numerical checks](verification.txt).

With default one-card flags, 128 interleaved full-vocabulary rows matched p9 bit for bit. The two-card raw checks matched 112 full-vocabulary rows across seven depths, through 128,064 tokens. [Per-case hashes and coverage](checks/evidence.tsv).

## Placement experiments

The initial budget allowed 278 us per token for 96 independent one-way 9 KiB peer writes across 48 attention and 48 expert visits, or 557 us for round trips. The selected implementation splits one 201,024-row padded output head. The peer computes 100,512 rows and writes narrow outputs directly into primary FP32 logits. The activation transfer took 5.1 us, and the two projection kernels averaged 151.9 and 150.6 us. Wide prefill uses the full primary head. [Exchange budget](BUDGET.md), [kernel profile](direct-profile-after.tsv).

Moving the drafter to the other GPU slowed both exact ABBA pilots. Each verify round needs the draft IDs, and each new draft needs the committed features. Expert sharding pays a synchronization cost on each routed visit. Isolated head-sharded attention was slower at 1K (15.1 to 27.0 us) and faster at 64K (816.7 to 506.4 us), with 3,072 matching floats on every leg. The mixed whole-layer candidate diverged from same-context p9 plain at generated token 43, so its proposed 327,680-token slot failed qualification. [Drafter pilots](placement.txt), [attention by depth](attention-heads.tsv), [independent attention repeats](attention-repeat.txt), [mixed-layer mismatch](screen/balanced-fixed-short.comparison), [tagged experiments](EXPERIMENTS.md).

At the 07:30 EST cutoff, the post-residency replays of earlier development work had completed 15/25 matrix configurations, 32/41 numerical checks, 1/16 screening probes, and 7/16 capacity comparisons. Pending historical runs and incomplete timing series are identified in the ledger. [Replay coverage](replay/README.md), [historical timing series](replay/paired.tsv).

## Staged deployment

The launcher configures one 131,072-token vision slot, target/draft microbatches of 512/512, adaptive cap 6, alias `inkling-mini`, port 8031 and the p9 fallback. Its syntax check passed. The staged server loaded its eight project libraries locally and matched p9 on six text replies and one real JPEG request, covering 2,688 generated IDs. Deployment remains with the orchestrator. The local evidence tag is `p11a-release`. For a later lab run, set `P11_UNTIL` to a future local deadline. [Manual launcher](../../live/deploy-p11a.sh), [library resolution](snapshot.ldd), [snapshot smoke test](snapshot-probes/audit.txt).

Sources include the [installed aikitoria P2P revision](https://github.com/aikitoria/open-gpu-kernel-modules/tree/0d96f83bbb4669b17f5978c3d5eaef23f3e5415f), [installed zagitta gpu-unlock revision](https://github.com/zagitta/gpu-unlock/tree/7eb51bccae0a0b7e599e084d971170e9a1e82d0c), [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [p9 verification scripts](../../p9/verify.sh).
