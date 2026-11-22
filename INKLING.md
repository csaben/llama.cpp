# Inkling 2 Mini on an RTX 3090

This branch (`inkling-3090`) is llama.cpp plus custom fused CUDA operators for Inkling 2 Mini's looped
mixture-of-experts architecture (`inkling2_loop`). On one RTX 3090 the Q6_K checkpoint reaches **159.3 tok/s** on tg128 and **152.4 tok/s**
in 700-token server generations at temperature zero. An experimental IQ4_NL dense model reaches
**174.2 tok/s** on tg128 and **167.6 tok/s** served, with a measurable change in greedy output.
Both use all three core passes, Q8_0 KV and F16 controllers.

Write-up: [*Looping on a 3090*](https://clark.saben.org/looping-on-a-3090).

**Just run it** (Linux, glibc ≥ 2.38, NVIDIA driver ≥ 570, an RTX 3090-class GPU; no CUDA toolkit or build):

```bash
uvx --from "git+https://github.com/csaben/llama.cpp@inkling-3090-v2#subdirectory=tools/inkling-3090" inkling-3090             # 2-bit, ~206 tok/s
uvx --from "git+https://github.com/csaben/llama.cpp@inkling-3090-v2#subdirectory=tools/inkling-3090" inkling-3090 --quality   # Q6_K, ~159 tok/s, near-exact
```

Models: [2-bit](https://huggingface.co/arelius/Inkling-2-Mini-Q2-3090-GGUF) ·
[Q6_K near-exact](https://huggingface.co/arelius/Inkling-2-Mini-Q6K-3090-GGUF) ·
[uncensored 2-bit](https://huggingface.co/arelius/Inkling-2-Mini-Uncensored-Q2-3090-GGUF). Prebuilt server:
[release inkling-3090-v2](https://github.com/csaben/llama.cpp/releases/tag/inkling-3090-v2).
Lab notebook with every step, number and profile: [`perf/RESULTS.md`](perf/RESULTS.md).

> **Agents:** start here. This file tells you how to build it, reproduce each number, find the core code and
> check a change. The upstream `AGENTS.md` / `CLAUDE.md` are ggml-org's contributor rules and apply only to
> upstream PRs.

## Results (llama-bench tg128, one RTX 3090, CUDA 12.8)

| config | tok/s | PPL (8×2048, `perf/ppl.txt`) | tag |
|---|---|---|---|
| stock llama.cpp (master + DSpark PR #32218) | 62.2 | 2.9381 | `acf48fa` |
| phase 1: fused ops, load-time weight concat | 124.1 | 2.9412 | `checkpoint-124tok` |
| phase 2: single-launch attention + router, Q8_1 activations | 140.7 | 2.9412 | `checkpoint-140tok` |
| + Q6_K dense projections/head, F16 loop controllers | 149.3 | 2.9407 | `variant-q6k-dense` |
| phase 3 variant, Q8_0 KV, consistent prefill/decode basis | 155.4 | 2.9404 | `da0965f` |
| Q6_K with batched messages, segmented projection and specialized residual mixing | 159.3 | 2.9404 | `a85419e` |
| Same kernels, experimental IQ4_NL dense projections and head | 174.2 | 2.9458 | `a85419e` |

The unfused reference (`LLAMA_INKLING_FUSED=0 LLAMA_INKLING_FUSED_QKV=0`) gives PPL 2.9401. The standard
error is 0.064.

## Reproduce

```bash
# 1. build (sm_86 = RTX 30-series; use your arch)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server llama-bench llama-perplexity llama-completion llama-quantize test-backend-ops

# 2. weights (17.6 GB)
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir ~/models/inkling-2-mini

# 3. speed: expect ~140 tok/s on a 3090
build/bin/llama-bench -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 3
# the same build with the fused paths off, for A/B (~64 tok/s):
LLAMA_INKLING_FUSED=0 LLAMA_INKLING_FUSED_QKV=0 build/bin/llama-bench -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 3

# 4. quality
build/bin/llama-perplexity -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -fa on -f perf/ppl.txt -c 2048 --chunks 8 -b 2048
MODEL=~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf GPU=0 perf/greedy.sh /tmp/greedy   # 3 prompts, temp 0

# 5. serve (OpenAI-compatible)
build/bin/llama-server -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 --port 8031
python3 perf/bench.py http://127.0.0.1:8031          # served tok/s (~128 exact, ~136 variant)
```

The fast variant: build its GGUF with the `llama-quantize` command in `perf/RESULTS.md` ("Phase 2 variants"),
then run with `LLAMA_INKLING_CTRL_F16=1`.

To reproduce the single-GPU decode measurements from the repository root:

```bash
export GPU=1 CUDA_VISIBLE_DEVICES=1 LLAMA_INKLING_CTRL_F16=1 IK_BATCH_MESSAGES=1
export IK_SEGMENTED_Q6=1 IK_SEGMENTED_IQ4=1 IK_MIX_SPECIALIZE=1
export MODEL="$HOME/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-q6k.gguf"
build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,512,1024 -r 5
python3 perf/server_run.py "$(mktemp -d)"
```

For the experimental model, use the sibling file `Inkling-2-Mini-IQ4_NL-dense-iq4nl.gguf`.
To regenerate it, run `bash perf/quantize-inkling-dense.sh ORIGINAL.gguf NEW.gguf` with the published
IQ4_NL GGUF as input. Each segmented kernel activates only for its matching weights on GA10x.
See the single-GPU results in `perf/RESULTS.md` for the measured quality differences.

The original configuration needs about 19 GB of VRAM at 32K context. For an OpenCode
experiment, use the exports above, select the desired GGUF with `MODEL`, and run:

```bash
build/bin/llama-server -m "$MODEL" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 16384 -np 1 --host 127.0.0.1 --port 8031 --alias inkling-mini
```

Use the OpenAI-compatible base URL `http://127.0.0.1:8031/v1` and model ID
`inkling-mini`. On the experimental IQ4 model, a chat-completions smoke test
returned correct arguments for automatic and required tool choice. A streamed
automatic call and the follow-up turn also reproduced the build status and
unique marker supplied by the tool. The checks exercised the HTTP protocol.
Vision remains unsupported on this branch.

## Experimental 200 tok/s server

The calibrated Q2_K gate/up and Q2_0 down model averaged 202.0 tok/s at
temperature zero and 201.0 at 0.7 in three 700-token requests per setting.
Its prefill perplexity was 3.0516 (reference 2.9401), outside the 0.5% quality
target. OpenCode 1.2.5 used its read tool and returned the correct file contents.
With a read-only tool configuration and 2,880 prompt tokens, its tool call and
reply decoded at 165.7 and 163.8 tok/s. With the default tool set, the initial
request used 16,963 prompt tokens. The 32K server read a fresh file and
returned its exact marker, decoding the two turns at 140.2 and 140.7 tok/s.

The live experiment uses a 65,536-token slot. GPU 1 had 10,236 MiB free after loading. From the repository root, start that configuration with:

```bash
export CUDA_VISIBLE_DEVICES=1 LLAMA_INKLING_CTRL_F16=1 IK_BATCH_MESSAGES=1
export IK_SEGMENTED_IQ4=1 IK_SEGMENTED_Q6=1 IK_MIX_SPECIALIZE=1 IK_CTRL_APPLY8=1
build/bin/llama-server -m "$HOME/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf" \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 1 --backend-sampling \
  --host 127.0.0.1 --port 8031 --alias inkling-mini
```

When the server is running, use this OpenCode invocation from any working directory:

```bash
OPENCODE_CONFIG=/home/arelius/llama.cpp-fast/perf/opencode-inkling.json opencode -m localinkling/inkling-mini
```

## Long context and prefill (phase 5)

Decode used to fall from 209 to 131 tok/s between an empty context and 32K tokens, because the attention over the
8 full-prefix layer visits was bound by its own arithmetic. That scan now runs on the tensor cores straight from the
Q8_0 cache, with no change to the cache format. The loop controllers of a batch are GEMMs. Two-bit model, one RTX 3090:

| | empty | 4K | 16K | 32K |
|---|---:|---:|---:|---:|
| decode before, tok/s | 209.5 | 168.7 | 148.9 | 131.2 |
| decode now, tok/s | 210.3 | 194.9 | 182.5 | 168.7 |

Prefill of 4,096 tokens went from 2,184 to 3,018 tok/s; a 16K-token prompt is answered after 6.3 s instead of 8.5 s.
Both are on by default; `IK_ATTN_MMA=0` and `IK_CTRL_GEMM_MIN=0` give the previous paths. Tables, method, what did
not work and the sources are in the phase 5 section of [`perf/RESULTS.md`](perf/RESULTS.md). The attention scan is
adapted from Project Maya's `maya-ik2` (MIT); see [`NOTICE.inkling`](NOTICE.inkling).

```bash
. perf/p5env.sh                                    # GPU 1, the kernel flags, $Q2 / $IQ4
build/bin/llama-bench -m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384,32768 -r 3
build/bin/llama-bench -m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512,4096,16384 -n 0
MODEL=$Q2 python3 perf/longctx.py --start --short -- --backend-sampling    # served: short, 16K and 32K prompts
MODEL=$Q2 python3 perf/spec_probe.py plain -- --backend-sampling           # code / edit / prose, plain or DSpark
```

## Speculative decoding as a window (phase 6)

With `IK_MR=1` a DSpark verify step of 2-8 tokens goes through the single-token fused kernels row by row, and each row's
logits are bit-identical to decoding the tokens one at a time (`perf/p6/mr_check.cpp`). With the cheaper drafter of the
same phase, DSpark is 1.19x plain decode on an edit of given code (235 vs 197 tok/s) and 1.10x at 16K context;
`LLAMA_DSPARK_ADAPTIVE=1` switches drafting off while it loses (new code and prose: 0.90x instead of 0.7x). The expert
matvecs are bound by dp4a work per token on this card, so a 4-row window costs 2.3 single tokens. Tables, what did not
work and the server command are in the phase 6 section of [`perf/RESULTS.md`](perf/RESULTS.md).

```bash
. perf/p5env.sh
IK_MR=1 LD_LIBRARY_PATH=build/bin build/bin/mr_check $Q2 perf/ppl.txt 300 48      # windows == single tokens, bit for bit
IK_MR=1 MODEL=$Q2 python3 perf/spec_probe.py dspark -- --backend-sampling --spec-type draft-dspark \
  --spec-draft-model $DRAFT --spec-draft-n-max 3 --spec-draft-p-min 0
```

## Core code

| file | what |
|---|---|
| `src/models/inkling2-loop.cpp` | the loop: 24 physical layers become 48 cache slots; core layers 6–17 run 3 passes; each pass reads the previous pass's bank |
| `src/models/inkling2.cpp` | the layer body: attention-residual mixes, the fused attention/router ops, the expert path, load-time weight concatenation (`load_fused`) |
| `ggml/src/ggml-cuda/inkling.cu` | the custom CUDA kernels: `res_mix(_norm)`, `shortconv_res`, `rel_bias`, `l2_clip`, `loop_ctrl_down/apply`, `moe_sum`, `inkling_attn`, `inkling_router` |
| `ggml/src/ggml-cpu/ops.cpp` | CPU reference implementations (what `test-backend-ops` compares against) |
| `tests/test-backend-ops.cpp` | test cases for every new op |
| `ggml/src/ggml-cuda/mmvq.cu` | the quantized matvec (2 rows/block for Q8_0/IQ4_NL, Q8_1 activations) |
| `perf/` | `RESULTS.md` (notebook), `bench.py`, `greedy.sh`, `kernsum.py` (nsys summary), `ppl.txt` |
| `perf/p2p/` | patched NVIDIA driver install for GPU↔GPU P2P on consumer cards (for the 2-GPU split) |

Switches (all on by default): `LLAMA_INKLING_FUSED`, `LLAMA_INKLING_FUSED_QKV`, `LLAMA_INKLING_Q8_ACT`,
`LLAMA_INKLING_ATTN_OP`, `LLAMA_INKLING_ROUTER_OP`, `LLAMA_INKLING_MOE_OP=0..3`. Opt-in: `LLAMA_INKLING_CTRL_F16=1`.

## Gates for any change

For Q8 KV, check a batched prefill followed by single-token decode. The fused
attention kernel uses unrotated keys and values. Versions before `156a872`
could pass both perplexity checks below yet generate unrelated text after a
batched prompt: prefill rotated the cache and fused decode interpreted it in
the original basis. Use fresh servers for this comparison:

```bash
export GPU=1 LLAMA_INKLING_CTRL_F16=1
export MODEL="$HOME/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-q6k.gguf"
probe=$(mktemp -d)
LLAMA_ATTN_ROT_DISABLE=1 LLAMA_INKLING_ATTN_OP=0 python3 perf/server_run.py "$probe/reference" --kv-probe --no-bench
python3 perf/server_run.py "$probe/fused" --kv-probe --no-bench
cmp "$probe/reference/gen1.txt" "$probe/fused/gen1.txt"
```

1. `build/bin/test-backend-ops -o <OP>` for every op you touch (add cases for new ops, with a CPU reference).
2. PPL within ~0.5% of the unfused 2.9401. Also check the decode-path PPL (`-ub 1 -c 1024 --chunks 2`),
   because decode takes different kernels than prefill.
3. `perf/greedy.sh`: divergence from the reference is fine only at near-ties (top-2 within ~0.1 logprob).
4. A served check with `perf/bench.py`. Record the step, tok/s, PPL and commit in `perf/RESULTS.md`.

## Where the time goes now, and what's next

7.1 ms per token: weight matvecs 4.7 ms (about 88% of the 3090's measured 891 GB/s) plus about 2.4 ms of
latency-bound glue (attention, router, mixes, loop controllers) at about 50 µs per layer pass. The ceiling from
the bytes read per token (4.12 GB) is 216 tok/s. That is the exact-weight model; the 2-bit model reads 2.42 GB per token
(about 369 tok/s).

Open work:
- **Glue megakernel** (in progress): persistent kernels that keep the per-pass glue on-chip and overlap it with
  weight streaming. The target is about 0.7 ms of glue, which gives about 185 tok/s exact and 200+ on the variant.
- **Two-GPU tensor parallel** with the P2P driver (`perf/p2p/`): about 96 all-reduces per token. Estimated
  ~320–380 tok/s once the glue is small, because the glue does not shrink when the model is split.
- DSpark speculative decoding pays on edits only (phase 6 above); a cheaper drafter head and tensor-core expert
  matvecs are the open pieces.
