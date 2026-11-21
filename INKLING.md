# Inkling 2 Mini on an RTX 3090

This branch (`inkling-3090`) is llama.cpp plus custom fused CUDA operators for Inkling 2 Mini's looped
mixture-of-experts architecture (`inkling2_loop`). On one RTX 3090 the Q6_K checkpoint reaches **159.3 tok/s** on tg128 and **152.4 tok/s**
in 700-token server generations at temperature zero. An experimental IQ4_NL dense model reaches
**174.2 tok/s** on tg128 and **167.6 tok/s** served, with a measurable change in greedy output.
Both use all three core passes, Q8_0 KV and F16 controllers.

Write-up: *Looping on a 3090* (link in the repo description).
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

Needs about 19 GB of VRAM at 32K context. Text only: vision and `tools` requests are broken upstream for this
architecture at this llama.cpp version.

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
the bytes read per token (4.12 GB) is 216 tok/s.

Open work:
- **Glue megakernel** (in progress): persistent kernels that keep the per-pass glue on-chip and overlap it with
  weight streaming. The target is about 0.7 ms of glue, which gives about 185 tok/s exact and 200+ on the variant.
- **Two-GPU tensor parallel** with the P2P driver (`perf/p2p/`): about 96 all-reduces per token. Estimated
  ~320–380 tok/s once the glue is small, because the glue does not shrink when the model is split.
- DSpark speculative decoding does not pay off for this MoE. See the "Batched (speculative verify)" section of
  RESULTS.md.
