# Inkling 2 Mini single-stream decode optimization (RTX 3090, GPU1)

Model: ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf (inkling2_loop, 26B MoE 128x8, 24 layers, layers 6-17 looped 3x)
Build: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF
Bench: CUDA_VISIBLE_DEVICES=1 build/bin/llama-bench -m <model> -ngl 99 -fa 1 -p 0 -n 128 -r 3
PPL:   CUDA_VISIBLE_DEVICES=1 build/bin/llama-perplexity -m <model> -ngl 99 -fa on -f perf/ppl.txt -c 2048 --chunks 8 -b 2048
Greedy: perf/greedy.sh <outdir>  (3 prompts, temp 0, 200 tokens)

| step | commit | tg128 tok/s | PPL | notes |
|---|---|---|---|---|
| baseline (nvcc 12.0 build) | acf48fa | 61.53 | - | user's number |
| baseline (nvcc 12.8 build) | acf48fa | 62.15 | 2.9381 | GGML_CUDA_GRAPH_OPT=1: 61.86; GGML_CUDA_GRAPH_CHUNK_SIZE=0: 61.49 (no gain) |
| step 1: fused ggml ops (res_mix, l2_clip, shortconv_res, rel_bias, loop_ctrl_down/apply, situ/siglu GLU, MoE sum via mul_mat) | 7889cdf, 82fb8d7 | 85.14 | 2.9363 | greedy: gen1 identical, gen2 diverges after 35 tok, gen3 after 15 tok (near-tie flips, see below) |
| step 1b: res_mix/loop_ctrl_down/rel_bias kernel tuning | fe7e6f8 | 93.06 | - | kernel-only |
| step 2: shortconv_res stores the conv state itself (no concat+cpy per convolution) | 94cb891 | 98.30 | 2.9400 | decode-path PPL (ub=1,c=1024,2 chunks): 4.0631 vs unfused 4.0660 |
| step 3: CUDA topk-moe fusion accepts the scale_bias (sum+eps) normalization -> routing tail fused | 053fc9a | 100.50 | - | test-backend-ops TOPK_MOE 692/692 |
| step 4: attention input projections fused at load time into one Q8_0 matrix per layer (1 matmul for SWA layers, 3 for loop/full-attn layers); originals are views into it (no extra VRAM) | a1803a1 | 107.57 | 2.9436 | decode-path PPL 4.0707 (unfused 4.0660); LLAMA_INKLING_FUSED_QKV=0 disables |
| step 5: faster res_mix (float4), l2_clip, rel_bias kernels | 7de2bc4 | 113.44 | - | kernel-only |
| step 6: K clip / V norm fused into the conv op, moe_sum op (expert weighted sum in 1 kernel) | a13cdce | 116.95 | 2.9410 | decode-path PPL 4.0633 (unfused 4.0660); gen1 identical to baseline, gen2/3 diverge after ~50 tokens; 1806 launches/token, GPU 8.6 ms/token of which mmvq 5.0 |
| step 7: q/k/v loop controllers stacked at load and batched (1 down + 1 apply+norm op per loop layer instead of 3+3+3 kernels) | 086080d | 120.77 | 2.9412 | decode-path PPL 4.0659 (unfused 4.0660); gen2 matches baseline for 863 chars |
| step 8: mmvq 2 rows/block for Q8_0/IQ4_NL (generic table), fused conv reads the recurrent state cells directly (no get_rows) | ec076be | 124.14 | 2.9412 (decode-path 4.0659) | mmvq micro-bench: 4-11% faster on 1k-9k row matvecs and expert matvecs; test-backend-ops MUL_MAT 2692/2692, MUL_MAT_ID 1077/1077 |

## Baseline decode graph (per token, 1 stream)
7764 non-view ggml nodes/token -> ~6100 kernel launches + ~1070 matvecs. Op histogram:
MUL_MAT 977, ADD 872, MUL 858, SCALE 629, CONCAT 548, RMS_NORM 514, CONT 480, CPY 337, TANH 289, FILL 240,
SUM_ROWS 192, DIV 192, SSM_CONV 192, GET_ROWS 146, CLAMP 145, SQR 144, SQRT 144, SUB 144, SIGMOID 144,
REPEAT 120, SOFT_MAX 120, SET_ROWS 96, MUL_MAT_ID 96, PAD 48, FLASH_ATTN 48, ARGSORT 48.

## Server-level results (perf/bench.py, llama-server -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1, port 8041)
| config | gen tok/s (mean of 3 prompts, 700 tok each) | notes |
|---|---|---|
| baseline (user's measurement) | ~58 plain / ~66 with DSpark | |
| ec076be plain | 115.7 @T=0, 115.4 @T=0.7 | perf/ref/server_plain |
| ec076be + DSpark n_max=3 | 96.2 @T=0, 89.9 @T=0.7 | acceptance ~45-57% on chat prompts, 15-20% on raw completions; the batched verify of a MoE costs more than it saves now |
| ec076be + DSpark n_max=2 | 96.6 @T=0, 89.5 @T=0.7 | |
| ec076be + DSpark n_max=1 | 87.0 @T=0, 84.6 @T=0.7 | |

## Final decode profile (ec076be, nsys node-level, llama-bench tg64)
GPU kernel time 8.11 ms/token, 1614 launches/token (baseline: 16.5 ms, ~7200 launches). mmvq 4.76 ms (59%; weight streaming
floor ~4.7 ms at 850 GB/s for 3.97 GB/token), res_mix 0.45, quantize 0.33, l2_clip 0.25, loop ctrl 0.46, FA 0.31,
rel_bias 0.20, rms_norm 0.27, shortconv 0.36, topk 0.17, router 0.16, situ/siglu 0.14, set_rows 0.13, moe_sum 0.08.
See perf/ref/final_profile.txt.

DSpark correctness: fused and unfused (LLAMA_INKLING_FUSED=0 LLAMA_INKLING_FUSED_QKV=0) DSpark runs diverge from the plain
greedy output at the same points (gen1: 100 vs 98 chars, gen2: 72 vs 76, gen3: 72 vs 72) -> the divergence comes from the
batched-verify numerics of speculative decoding itself (present in the unfused reference too), not from the fused rollback-slot
state store. The slot layout written by the fused op (slot k = inputs as of k tokens back, plane stride size*row_size) was
checked against build_conv_state_store line by line.

## What changed (summary)
1. New ggml ops (CPU reference + CUDA + test-backend-ops tests), used only by the inkling2 graph:
   - L2_CLIP (row L2 clip with optional broadcast prescale), RES_MIX (attention-residual mix over up to 9 states),
     SHORTCONV_RES (causal short conv + residual + dynamic taps, stores the recurrent state incl. rollback slots itself,
     optional per-head rms_norm / l2_clip, optional direct read of the state cells by s_copy ids),
     REL_BIAS (relative-position bias gather + sinks + mask -> per-head F16 attention mask),
     LOOP_CTRL_DOWN / LOOP_CTRL_APPLY (loop-pass message controllers, batched over q/k/v, optional fused norm),
     MOE_SUM (weighted expert sum), GLU ops SITU and SIGLU.
2. CUDA: topk-moe fusion also covers the scale_bias (sum+eps) normalization; mmvq uses 2 rows/block for Q8_0/IQ4_NL.
3. Model: load-time concatenation of the attention input projections into one Q8_0 matrix per layer (q|rel|sink|gate|k|v
   [|conv proj]); the file tensors become views, no VRAM duplication. q/k/v loop controllers stacked the same way.
   Unfused graph available with LLAMA_INKLING_FUSED=0 (ops) and LLAMA_INKLING_FUSED_QKV=0 (weights) for A/B checks.
4. Decode graph: 7764 -> ~1640 kernel launches per token; GPU time 16.5 ms -> ~8.0 ms per token.

## Deploy
Build:
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF
  cmake --build build -j 12 --target llama-server llama-bench llama-perplexity llama-completion
Serve (plain decode, recommended):
  CUDA_VISIBLE_DEVICES=1 build/bin/llama-server -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --host 127.0.0.1 --port 8031

## Risks / known issues
- Greedy outputs differ from the baseline build after ~15-200 tokens on 2 of 3 prompts (near-tie flips amplified by MoE
  routing); PPL identical within noise (2.9412 vs 2.9381, stderr 0.064; decode-path PPL 4.0659 vs 4.0660).
- The fused ops bypass LoRA: the graph falls back to the plain ops when any LoRA adapter is loaded.
- shortconv_res writes the recurrent state as a side effect through its src[5] view (like ggml_cpy/set_rows targets).
- The fused weights are assembled from the file at load (direct file reads, works with and without mmap); a memory-fit
  (no_alloc) pass loads the plain tensors instead.
- No speculative decoding gain anymore: DSpark is slower than plain decode at this speed.
- Only the CUDA backend has fast kernels; the CPU implementations are reference-quality (correct, not optimized).

# Phase 2 (start: checkpoint-124tok = 5196d97, 124.4 tok/s tg128)

## Budget re-derived from the GGUF (per decode token)
experts 1847 MB (8/128 per pass, IQ4_NL layers 0-19, Q8_0 layers 20-23), dense 1781 MB (Q8_0 1373 MB + F32 409 MB:
loop ctrl down 170 MB, gain/forget 170 MB, router 57 MB, bias tables 3 MB), output head 492 MB -> 4121 MB/token.
Measured on GPU1: raw read bandwidth 891 GB/s (1 GiB), 737 GB/s for a 2 MB read, 711 GB/s for 8 MB, 831 GB/s for 32 MB
(ramp/tail of each kernel). Empty kernel in a CUDA graph: 0.65 us; tiny 82x256 kernel: 1.10 us.
Floor at 891 GB/s: 4.62 ms = 216 tok/s (weights only, exact types). 200 tok/s = 5.0 ms.

## Where the 8.0 ms go (nsys, one steady-state token, 1613 launches)
| part | time | bytes | GB/s |
|---|---|---|---|
| mmvq expert gate_up (IQ4_NL, 23.9 MB) | 29.7 us x 44 | | 805 |
| mmvq expert down (IQ4_NL, 11.9 MB) | 15.7 us x 44 | | 760 |
| mmvq attn proj SWA (8600 rows Q8_0, 21.0 MB) | 25.9 us x 40 | | 813 |
| mmvq attn proj full-attn q group (16.0 MB) / k, v groups (2.6 MB) | 20.2 / 4.7 us | | 794 / 557 |
| mmvq wo (7.5 MB) | 10.4 us x 48 | | 723 |
| mmvq output head (492 MB) | 557 us | | 883 |
| all mmvq | 4.74 ms | ~3700 MB | 780 avg |
| F32 loop ctrl down/apply (340 MB) + router mul_mat_vec_f (57 MB) | 0.61 ms | 409 MB | 670 |
| flash attention (vec + combine) | 0.30 ms | | |
| ~1260 small kernels (res_mix, rms_norm, quantize, l2_clip, rel_bias, shortconv, set_rows, topk, situ, moe_sum...) | 2.33 ms | | |

So: weight streaming is at 780 GB/s on average (88% of measured peak); the other 3 ms are ~1300 latency-bound launches
of 1.2-4 us each. A persistent megakernel would replace a kernel boundary (~1.1 us in a graph) by a grid barrier
(~1.5-3 us on 82 SMs with cooperative groups) -> no gain per se; the gain is in the number of dependency points.
Plan: fuse the per-layer body into ~8 launches (prologue/epilogue fusion around the matvecs, one attention kernel),
then tune the matvecs (bytes in flight, tail), then quality-gated byte reduction and the DSpark verify path.

## Phase 2 steps
| step | commit | tg128 tok/s | PPL | notes |
|---|---|---|---|---|
| p2-2: GGML_OP_INKLING_ATTN: the single-token attention block (K/V conv + per-head norm/clip + cache store + Q/R clips + relative bias + attention with the zero sink + sigmoid gate, Q8_1 output) in one kernel (8 kv-head x 8 cell-split blocks, last block combines); MoE: merged gate\|up matvec + SITU, down matvec, fused weighted sum + FFN conv + residual (INKLING_MOE_SUM_CONV); SITU added to the fused mmvq GLU; custom MoE matvec kernels (INKLING_MOE_UP/DOWN) kept opt-in (LLAMA_INKLING_MOE_OP=2, slower than mmvq) | see tag checkpoint-134tok | 134.1 | 2.9421 (decode-path 4.0482, unfused 4.0660) | 801 launches/token; attention kernel 13 us (replaced ~24 us of 11 kernels); greedy gen1 614/614 and gen2 903/903 identical to step 8, gen3 70. test-backend-ops: INKLING_ATTN 4/4 (+8 via CONCAT), INKLING_MOE_* 4/4 (+6), MUL_MAT 2696/2696, MUL_MAT_ID 1081/1081 |
| p2-4: router op packs norm\|bias and writes the Q8_1 expert activation, SITU writes Q8_1 (no quantize launches left), Q8_1 mul_mat_id keeps CUDA graphs enabled, F16 loop controllers as an opt-in variant (LLAMA_INKLING_CTRL_F16=1: 141.9 tok/s, PPL 2.9380 / decode 4.0532) | checkpoint-140tok | 140.7 | 2.9412 (decode-path 4.0674) | 654 launches/token, 7.33 ms wall; served (llama-server, q8_0 KV, bench.py) 128.7 tok/s; greedy gen1 217, gen2 863, gen3 173 |
| p2-3: GGML_OP_INKLING_ROUTER: FFN mix + norm + router logits + sigmoid/top-k/weight normalization in one launch (16 blocks, last block selects); attention kernel loop: relative bias folded into the dot partial sums, transposed 4-head butterfly reduction, per-step softmax update | 3f99170+ | 137.7 | decode-path 4.0674 (unfused 4.0660) | 750 launches/token; greedy gen1 412, gen2 863, gen3 70 (routing near-ties move with the summation order) |
| p2-1: res_mix_norm (mix + rms norm in one op, F32 or Q8_1 output), loop_ctrl_apply with Q8_1 output, Q8_1 activations accepted as mmvq src1 (CUDA; CPU repacks to Q8_0); attention/FFN input norms fused into the mix; one projection matmul over the three attention inputs (ncols=3, weights read once) | 04c3ea5 | 128.9 | 2.9400 (decode-path 4.0595, unfused 4.0660) | 1493 launches/token; greedy gen1 412 (same), gen2 724, gen3 70 chars common with step 8 (near-tie flips). test-backend-ops RES_MIX 54/54, LOOP_CTRL_APPLY 15/15, MUL_MAT 2696/2696, MUL_MAT_ID 1081/1081 |

## Batched (speculative verify) steps: why DSpark cannot reach 200 tok/s here
llama-bench pp2 = 184 tok/s (10.8 ms per 2-token step), pp4 = 61-68 tok/s (~60 ms per 4-token step, Nov 21 build).
The MoE experts are routed per token, so a verify step of n tokens reads up to n x 1847 MB of expert weights (random
8-of-128 routing: ~3.6x for n = 4) while only the dense 1781 MB + head 492 MB are shared. Bytes per step at the bandwidth
floor: n=1 4.12 GB, n=2 5.97 GB, n=4 9.66 GB. With a per-token draft acceptance p, the expected accepted tokens per step
are 1 + p + p^2 + ..., so n=2 beats plain decode only for p > 0.45 (7% gain at p = 0.55) and n=4 needs p > 0.68; the
measured acceptance (phase 1) is 45-57% on chat prompts. The batched mul_mat_vec_q_moe kernel also runs at ~650 GB/s
effective (148 us for 4 tokens of gate_up) and the fused decode ops fall back to the unfused graph for n_tokens > 1.
Conclusion: speculative decoding is at best a few percent for this architecture on one GPU; the single-token path is
where the time is.

## Phase 2 variants (byte reduction, quality-gated)
| variant | how | tg128 tok/s | PPL (c=2048, 8 chunks) | decode-path PPL | notes |
|---|---|---|---|---|---|
| exact (default) | Inkling-2-Mini-IQ4_NL.gguf as published | 140.7 | 2.9412 | 4.0674 | the reference; unfused 2.9401 / 4.0660 |
| F16 loop controllers | `LLAMA_INKLING_CTRL_F16=1` (the F32 controller weights, 340 MB/token, are converted to F16 at load) | 141.9 | 2.9380 | 4.0532 | opt-in at run time, no model change |
| Q6_K dense | `Inkling-2-Mini-IQ4_NL-dense-q6k.gguf`: attention projections and the output head requantized Q8_0 -> Q6_K (llama-quantize --allow-requantize with per-tensor overrides, experts and embeddings untouched) | 145.6 | 2.9402 | 4.0704 | 16.16 GiB; dense bytes 1373 -> 1030 MB, head 492 -> 362 MB |
| Q6_K dense + F16 controllers | both | 149.3 | 2.9407 | - | served 136.0 tok/s (bench.py, q8_0 KV); exact model served 128.2 |
All PPL values are within the 0.064 standard error of each other.

Variant GGUF:
  build/bin/llama-quantize --allow-requantize --output-tensor-type q6_k --token-embedding-type q8_0 \
    --tensor-type 'attn_q=q6_k' --tensor-type 'attn_k=q6_k' --tensor-type 'attn_v=q6_k' --tensor-type 'attn_output=q6_k' \
    --tensor-type 'attn_gate=q6_k' --tensor-type 'attn_rel=q6_k' --tensor-type 'attn_sink_proj=q6_k' \
    --tensor-type 'attn_k_conv_proj=q6_k' --tensor-type 'attn_v_conv_proj=q6_k' \
    --tensor-type 'blk\.(1[0-9]|[0-9])\.ffn_.*_exps=iq4_nl' --tensor-type 'blk\.2[0-3]\.ffn_.*_exps=q8_0' \
    ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-q6k.gguf iq4_nl

## Phase 2 summary
Start: 124.4 tok/s (8.04 ms/token, 1614 launches). End: 140.7 tok/s exact (7.11 ms, 654 launches); 149.3 tok/s with the
two quality-gated variants (Q6_K dense projections/head + F16 loop controllers, PPL unchanged within noise).
Served (llama-server, q8_0 KV, perf/bench.py, 700-token generations): see the rows above (the server adds ~0.7 ms/token).

What the decode graph looks like now (per layer pass, 48 passes/token): attention input mix+norm -> Q8_1 (1 launch, or the
loop controllers: 2-3 launches), one attention-projection matvec (all three inputs, weights read once), the attention block
(1 launch: K/V conv + norm/clip + cache store, Q/R clips, relative bias, softmax with the zero sink, gate, Q8_1 output),
wo matvec, wo conv+residual (1), FFN mix+norm+router+top-k (1, Q8_1 side output), gate|up matvec, SITU -> Q8_1 (1),
down matvec, weighted sum + FFN conv + residual (1). 13-14 launches per pass instead of 34.

## Where the remaining 7.1 ms go (exact model, checkpoint-140tok, nsys)
| part | ms/token | note |
|---|---|---|
| expert matvecs (gate\|up 1152x8 + down 576x8, mmvq) | 2.34 | 1847 MB at ~790 GB/s |
| attention projection matvecs (4300 / 4332 blocks) | 1.26 | 1370 MB (Q8_0) at ~800 GB/s |
| output head matvec | 0.56 | 492 MB at 883 GB/s |
| wo matvecs | 0.54 | 361 MB at 720 GB/s (small matrices: ramp/tail) |
| fused attention kernel | 0.63 | 13 us x 48; ~6 dependent memory round trips + a split combine per launch |
| fused router kernel | 0.54 | 11 us x 48; the mix+norm chain, 1.2 MB of F32 router weights per layer, the top-k |
| attention-input mix+norm, message mixes | 0.38 | single-block latency-bound kernels (4 us each) |
| loop controllers (F32: 340 MB) | 0.55 | 24 looped passes; F16 variant: 0.43 |
| conv+residual, SITU, misc small kernels | 0.30 | |
| launch gaps | 0.25 | CUDA graph |
Weights-only floor at the measured 891 GB/s: 4.62 ms (216 tok/s) exact, 4.0 ms with the Q6_K/F16 variants (250 tok/s).
The matvecs run at 88% of that peak (ramp/tail of 240 dependent launches per token); the fused glue is ~2.4 ms of
latency-bound work (ten kernels per layer, each a chain of 3-8 dependent memory round trips at ~0.7 us).

## 200 tok/s: what it would take
5.0 ms/token. With exact weights the matvecs alone need 4.6 ms at peak bandwidth, leaving 0.4 ms for 48 layers of
attention, routing, norms and convolutions: not reachable with this kernel structure on one RTX 3090. With the Q6_K dense
variant the floor is 4.0 ms; 200 tok/s would then need the glue at ~1 ms (about 2 us per layer pass, i.e. a persistent
per-layer megakernel in which the attention/router/norm phases overlap with the weight streams) plus the matvecs at
>= 95% of peak. Speculative decoding does not help here (the per-token expert routing means n draft tokens read n times
the expert bytes; see the batched-step analysis above).

## Deploy
Build:
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF
  cmake --build build -j 12 --target llama-server llama-bench llama-perplexity llama-completion llama-quantize
Serve, exact weights (140.7 tok/s tg128):
  CUDA_VISIBLE_DEVICES=1 build/bin/llama-server -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --host 127.0.0.1 --port 8031
Serve, fast variant (149.3 tok/s tg128; PPL 2.9407 vs 2.9412):
  LLAMA_INKLING_CTRL_F16=1 CUDA_VISIBLE_DEVICES=1 build/bin/llama-server -m ~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-q6k.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --host 127.0.0.1 --port 8031
Switches (all default on): LLAMA_INKLING_FUSED=0 (phase-1/2 ops off), LLAMA_INKLING_FUSED_QKV=0, LLAMA_INKLING_Q8_ACT=0,
LLAMA_INKLING_ATTN_OP=0, LLAMA_INKLING_MOE_OP=0|1|2|3 (3 = default), LLAMA_INKLING_ROUTER_OP=0; LLAMA_INKLING_CTRL_F16=1 (opt-in).

## Risks / known issues (phase 2)
- The fused decode ops apply to single-token, single-sequence ubatches only (n_tokens == 1, n_seqs == 1); batched steps
  (speculative verify, several slots in one ubatch) use the phase-1 graph. Q8_1 activations are used for 3*n_tokens <= 8.
- The fused attention op writes the K/V cache and the recurrent state as side effects (like the phase-1 conv op); the
  router op writes the expert ids/weights and the Q8_1 activation into side tensors; the graph keeps those ops alive
  explicitly (ggml_build_forward_expand).
- Greedy outputs differ from the phase-1 build after 70-860 characters on the three test prompts (near-tie flips from
  summation-order changes; PPL and decode-path PPL are unchanged within noise).
- The last-block-combines kernels (attention, router) use persistent arrival counters allocated once per CUDA context.
- The CPU backend has reference implementations of every new op (test-backend-ops compares against them), but they are not optimized.
- The custom MoE matvec kernels (INKLING_MOE_UP/DOWN) are correct but slower than ggml's mmvq; they are opt-in (LLAMA_INKLING_MOE_OP=2).
- The Q6_K variant's attention projection matvec runs at ~660 GB/s (the Q6_K dot product is compute-heavier); a tuned
  Q6_K mmvq configuration for Ampere would add a few percent.

## Phase 3 handoff and mixed-prefill validation (2026-11-21)

`66678ff` joins the five local kernel commits through `99c35a6` with the fork's
reproduction guide. The projection megakernel and router+convolution remain
opt-in. Fable measured the router v2 at ~8 µs (from 11.8) and attention v2 at
10.7 µs (from 12.5); the four-row Q6_K matvec brought the variant to ~154 tok/s.

At `156a872`, an 827-token batched prefill exposed a cache-basis mismatch.
Ordinary attention used Hadamard-rotated Q8 KV; the fused decode operator used
raw Q8 KV. The first generated token came from prefill and matched, but the
second token could be unrelated to the prompt. Disabling fused attention,
using F16 KV, using single-token prefill, or disabling KV rotation eliminated
the observed failure. Mixed prefill/decode with Q8 KV now uses one basis
throughout. See the serving regression probe in `INKLING.md`.

On GPU 1 (RTX 3090), Q6_K dense, F16 controllers, Q8_0 K/V, compare
`IK_ATTN_CELLS_PER_SPLIT=64` against the new default. Command:
`llama-bench -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,512,1024 -r 3`
with the same variant model. Both columns use the corrected cache basis.

| Prefix depth | Fixed 64 (tok/s) | Adaptive (tok/s) |
|---:|---:|---:|
| 0 | 154.72 ± 0.29 | 153.91 ± 0.34 |
| 512 | 145.06 ± 0.14 | 148.87 ± 0.28 |
| 1024 | 140.61 ± 0.17 | 145.86 ± 0.22 |

The adaptive rule uses 128 cells per split only on GA10x with Q8_0 KV and a
KV extent of 513–2304. Q8 served throughput (`perf/server_run.py`, localhost,
16K context, one slot) at T=0 / 0.7 was 147.7 / 147.2 fixed and 148.7 / 148.6
adaptive. These are decode rates. The short-context benchmark and the longer
served generations exercise different context lengths.

`da0965f` also sums the selected router weights in rank order. The three
200-token top-1 completions then matched byte-for-byte across fixed/adaptive
splits and a fresh server restart. The 827-token mixed-prefill probe matched
standard unrotated attention for all 12 generated tokens. CUDA vs CPU:
21/21 attention and router cases passed, including the opt-in convolution
reading and writing the same recurrent cell. Quality on `perf/ppl.txt`:
2.9404 ± 0.0640 (8 × 2048 prefill) and 4.0640 ± 0.2878 (2 × 1024, ubatch 1).
Final tg128 at depths 0 / 512 / 1024: **155.40 / 150.08 / 146.95 tok/s**.
Final served means: **149.5 tok/s at T=0, 149.1 at T=0.7** (three 700-token
generations per temperature). Reaching 200 tok/s at depth zero still requires
cutting about 1.44 ms from the 6.44 ms decode step.

## Single-GPU pass-message batching and segmented projections (2026-11-21)

All runs used one RTX 3090, all three core passes, Q8_0 K/V, F16 controllers
and eight routed experts. The server numbers are mean decode rates for three
700-token generations per temperature, one slot and a 16K context. See
`INKLING.md` for the commands. Neither of the following optimizations needs a
change to the checkpoint's weights.

- `81f37bf` batches the 12 independent sets of message queries against the
  previous pass's state bank. The trace drops from 654 to 632 launches per token.
- `2865762` (IQ4_NL) and `866e39f` (Q6_K) compute only the projection column
  consumed by each attention row. The exact version accumulates four partials
  in registers in the order used by GA10x MMVQ. The faster trial that changed
  the addition order reached 174.6 tok/s but changed greedy generations.

| Dense weights | tg128 | Served T=0 / 0.7 | Prefill PPL, 8 × 2048 | Decode PPL, 2 × 1024 |
|---|---:|---:|---:|---:|
| Prior Q6_K checkpoint | 155.4 | 149.5 / 149.1 | 2.9404 | 4.0640 |
| Q6_K, batched messages + segmented projection | 159.0 | 152.4 / 151.7 | 2.9404 | 4.0640 |
| IQ4_NL, batched messages + segmented projection | 173.7 | 165.9 / 165.5 | 2.9458 | 4.0990 |
| IQ4_NL projections with Q6_K head | 165.5 | — | 2.9429 | 4.0781 |

The Q6 kernels reproduced all three 200-token greedy outputs from the
155.4 checkpoint. The IQ4 kernels reproduced all three from the same IQ4 model
with both optimizations off. Both paths reproduced their respective 12-token
reference continuations after the 827-token batched prefill. The backend suite
passed 23 attention/router cases. The CPU backend skipped two additional
Q6_K × Q8_1 projected-attention cases because it lacks that input combination.

Changing dense quantization changes the model. Against Q6_K, the IQ4 model's
first differing tokens on the three greedy prompts were 40, 50 and 30
(one-based). The reference top-two logprob margins were 0.431, 0.528 and
0.211. Each exceeds the branch's 0.1 near-tie tolerance. The mixed-head trial
also exceeded the tolerance. Keep Q6_K for the checkpoint-quality path.
The IQ4 candidate preserves the source's expert mixture, including Q8_0
experts in layers 20–23. Its recipe is `perf/quantize-inkling-dense.sh`.

Other trials on the IQ4 model included 16/32-lane expert matvecs, four routed
row tile sizes (1/2/4/8), a 128-thread router, fused router/convolution and
skipping fully masked cache tiles. None beat the retained configuration across
the paired tests. The trace of the first segmented kernel still spent 2.33 ms
per token on expert matvecs, 0.55 ms on attention and 0.35 ms on routing.

### Specialized residual mixing and final checks (`a85419e`)

`IK_MIX_SPECIALIZE=1` compiles the one-token residual mix with a fixed
state count for one through five states. Validation also exposed an existing
indexing bug in which CUDA treated the ninth state as the normalization
weight. All 12 affected cases failed with specialization disabled before the
fix. The corrected CUDA code counts states separately from the normalization
weight and declines graphs beyond its nine-state capacity. The CPU supports
the ten-state fallback case.

| Dense weights | tg128, depth 0 | Depth 512 | Depth 1024 | Served T=0 / 0.7 | Decode PPL |
|---|---:|---:|---:|---:|---:|
| Q6_K | 159.26 ± 0.17 | 153.91 ± 0.19 | 150.32 ± 0.26 | 152.4 / 152.1 | 4.0640 |
| IQ4_NL | 174.19 ± 0.25 | 167.69 ± 0.42 | 163.91 ± 0.15 | 167.6 / 167.1 | 4.0990 |

Each final row used five `llama-bench` repetitions. The Q6 row matched all
three 200-token greedy outputs of the 155.4 checkpoint, and the IQ4 row
matched its own three pre-optimization baselines. Both also matched their
respective 827-token prefill probes for all 12 generated tokens. The expanded
CUDA/CPU suite passed 93 supported residual-mix, router and attention cases,
including whole-graph IQ4 projection tests. Two Q6 projected-attention
cases lack CPU support. The historical prefill command reproduced 2.9404
(Q6_K) and 2.9458 (IQ4_NL). With explicit `-ub 2048`, eight-chunk prefill PPL was
2.9384 (Q6_K) and 2.9479 (IQ4_NL). The faster experimental row requires
a further 0.74 ms reduction per token to reach 200 tok/s at depth zero.
