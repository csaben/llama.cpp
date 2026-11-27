# Phase 11, Stream A

The frozen nine-file package selects the direct peer-write output head. The selected source is `a16c61e5`, with a 131,072-token vision slot, uncensored Q2, folded shortlist drafter, 512/512 microbatches and p9 fallback. The two-card candidate measured 245.84 tok/s (+4.84 against 241) on zero-depth plain decode and 355.20 tok/s (+4.20 against 351) on the short DSpark edit. [Manifest](snapshot.sha256), [manual launcher](../../live/deploy-p11a.sh).

| Uncensored Q2, tok/s | One-card p9 | Split output head | Change |
|---|---:|---:|---:|
| Plain decode at 0 | 239.93 | 245.84 | +2.46% |
| Plain decode at 4,096 | 220.51 | 225.51 | +2.27% |
| Plain decode at 16,384 | 208.03 | 212.40 | +2.10% |
| Plain decode at 32,768 | 191.47 | 195.41 | +2.06% |
| Plain decode at 65,536 | 161.83 | 164.23 | +1.48% |
| DSpark edit, 384 generated | 348.40 | 355.20 | +1.95% |
| DSpark edit, 3,000 generated | 295.60 | 301.10 | +1.86% |
| Prefill at 4,096 | 3,527.25 | 3,582.34 | +1.56% |
| Prefill at 16,384 | 3,218.00 | 3,258.62 | +1.26% |
| Prefill at 65,536 | 2,441.15 | 2,445.14 | +0.16% |
| Near-full prefill, 128,064 tokens | 1,708.43 | 1,693.26 | -0.89% |
| Near-full DSpark, 3,000 generated | 206.37 | 208.71 | +1.13% |

Each published row uses old/new/new/old after the unrelated service exited. The synthetic legs use three measurements per depth. The edit legs use the median of three uncached, fixed-length completions. Every served comparison above matched the independent p9 plain IDs and full message. [Synthetic legs](paired/quiet-batch/), [served legs](paired/quiet/), [physical card telemetry](telemetry.tsv).

| Capacity or quality | One-card p9 | Split output head |
|---|---:|---:|
| Largest verified vision slot at 512/512 | 253,952 | 253,952 |
| 128K server peak MiB, GPU1 | 18278 | 18278 |
| 128K server peak MiB, GPU0 | 0 | 416 |
| Uncensored PPL, pp / tg / MR | 3.0571 / 4.1915 / 4.1877 | 3.0571 / 4.1915 / 4.1877 |
| Calibrated PPL, pp / tg / MR | 3.0530 / 4.1979 / 4.2038 | 3.0530 / 4.1979 / 4.2038 |

Both servers completed 250,944 prompt tokens and 3,000 generated tokens, identical to one-card plain. Both encountered allocation failures at 262,144. The calibrated model is the source of the brief's reference perplexities. [Capacity and memory](capacity/), [model comparison](model-comparison.txt).

34/34 configurations, 194/194 cells, 360 drafted replies, 347,520 IDs plus complete messages equal. 32 additional full-vocabulary logit rows from the two-card build matched p9 bit for bit after the unrelated service exited. The default one-card build retained bitwise equality across 128 interleaved rows. The rollback/boundary checks verified 2,714 rows, plus the independent 217-row width 2-7 run at a 16,384-token prefix. Both backend suites passed, 64/64 IQ4 and 1,110/1,110 routed-operation cases. Six staged text replies and a real JPEG matched p9 on all 2,688 output IDs and the complete messages. [Release matrix](release-matrix/), [numerical results](checks/), [snapshot probes](snapshot-probes/).

The initial exchange budget was 278 us per token for 96 one-way 9 KiB transfers. The direct head instead replicates half the padded projection, computes 100,512 rows per card, and preserves each row's arithmetic. The primary/peer kernels averaged 151.9/150.6 us and the activation transfer took 5.1 us. Whole-drafter placement slowed both exact ABBA pilots because each speculative round consumes committed features and draft IDs. Replicated experts add synchronization between layers; the alternate implementations and their coverage are recorded separately. [Budget](BUDGET.md), [kernel profile](direct-profile-after.tsv), [experiments](EXPERIMENTS.md), [post-residency replays](replay/README.md).

The frozen package is `~/inkling-live-bin-p11a-a16c61e5`. The manual launcher uses port 8031, alias `inkling-mini`, and restores the p9 deployment on failure. All testing used the lab ports. Set `P11_UNTIL` to a future local deadline to reproduce a GPU test. [Provenance](provenance.txt), [linked libraries](snapshot.ldd).

[Installed P2P kernel](https://github.com/aikitoria/open-gpu-kernel-modules/tree/0d96f83bbb4669b17f5978c3d5eaef23f3e5415f), [installed gpu-unlock](https://github.com/zagitta/gpu-unlock/tree/7eb51bccae0a0b7e599e084d971170e9a1e82d0c), [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html), [p9 verification](../../p9/verify.sh).

Outstanding post-residency numerical checks at the time of this audit: logits-0-4096, logits-0-16384, logits-0-65528, deep-0-128064, logits-1-1, logits-1-4096, logits-1-16384, logits-1-32768, deep-1-65536, deep-1-128064.

## Additional qualification details

Plain decode exceeded 241 by 4.84 tok/s. Its 2.46% improvement over the paired p9 control is modest. The short DSpark edit cleared 351 by 4.20 tok/s. At 128K the split head consumed another 416 MiB on physical GPU0 and retained the same sampled peak of 18,278 MiB on physical GPU1. [Depth sweep](paired/quiet-batch/performance.tsv), [near-full memory and timing](capacity-paired.tsv).

At 8K on IQ4_NL dense, plain decode changed from 196.35 to 201.95 tok/s. The short edit changed from 246.15 to 245.00 tok/s. All twelve completions matched IQ4 p9 plain. [Four-bit ABBA](iq4-release/performance.tsv).

The model-shaped attention probe tested 4 depths with resident Q8 KV, 24 query heads and 8 KV heads. At 1,024 tokens, graph latency rose from 15.1 to 27.0 us. At 65,536 it changed from 816.7 to 506.4 us. Every tested leg matched all 3,072 output floats and independent CPU softmax samples. These figures describe the isolated attention graph. [Raw output](attention-heads.txt), [all depths](attention-heads.tsv).

The whole-layer candidate reached prompt evaluation at 327,680 tokens, then hit a mixed-expert graph assertion. After the tagged fix, its 8K short edit first diverged from same-context one-card p9 at token 43, so the larger slot failed qualification. [Compared IDs and message](screen/balanced-fixed-short.comparison).

The packaged server loaded all eight shared libraries from the staged directory. Six uncached edit/agent replies and one real JPEG request matched p9 plain on the complete messages and all 2,688 generated IDs. The JPEG used 238 prompt tokens and generated 384. [Text checks](snapshot-probes/staged.txt), [image request](capacity/snapshot-image.timings), [loader paths](snapshot.ldd).

The post-residency replay ledger records 15/25 matrix configurations, 22/41 numerical checks, 1/16 screening probes, 7/16 historical capacity comparisons. Each pending entry is identified separately. Only completed ABBA comparisons contribute reported performance. [Replay ledger](replay/README.md), [historical timing progress](replay/paired.tsv).

The local evidence tag is `p11a-release`. Set `P11_UNTIL` to a future local deadline when reproducing the GPU checks. The launcher was syntax-checked. Production deployment remains with the orchestrator.
