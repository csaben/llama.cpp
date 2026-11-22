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
(Correction, phase 6: 216 tok/s is the ceiling of the exact IQ4_NL weights only, 4.12 GB per token. The calibrated 2-bit
model that the live server runs reads 2.42 GB per token, a ceiling of about 369 tok/s; see the phase 6 section.)

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
(These floors are for the exact-weight files. The 2-bit model: 2.42 GB per token, 2.7 ms, about 369 tok/s.)
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

### DSpark on coding chats at the optimized checkpoint

Tested `a85419e` on GPU 1 with the dense IQ4_NL candidate and the flags in
`INKLING.md`. One slot, 16K context, Q8_0 K/V, temperature zero, seed 1,
prompt caching disabled, 384 generated tokens per request. Individual cells
are one run. DSpark used the bundled Q8_0 drafter on the same GPU and
`--spec-type draft-dspark`; each mode ran in a fresh server. Draft counts
are accepted/proposed.

| Coding chat | Plain tok/s | DSpark max 3 tok/s | Accepted | DSpark max 1 tok/s | Accepted |
|---|---:|---:|---:|---:|---:|
| ISO-8601 duration parser and tests | 170.71 | 109.74 | 224/475 | 99.83 | 157/225 |
| Review the fork's nine-state CUDA dispatch and write edge cases | 164.36 | 135.19 | 259/370 | 109.04 | 172/211 |
| Review the fork's quantization shell script for output-path races | 166.43 | 98.08 | 205/530 | 97.07 | 143/239 |
| Arithmetic mean | 167.17 | 114.34 | 688/1375 | 101.98 | 472/675 |

Acceptance alone did not predict a speedup. The one-draft configuration
accepted 70% of proposed tokens and still took longer on each coding chat.

### Follow-up single-GPU trials

With dense IQ4_NL on GPU 1, a two-stage channel-tiled residual mix passed all
70 supported cases. In a paired run it reached 173.97 ± 0.22 tg128 against
174.79 ± 0.22 for the existing kernel. At depth 1024 the rates were 164.10
and 164.16. Testing KV padding at 32, 64, 128, 256 and 512 tokens also
left the default 256 competitive. Two- and four-row router blocks passed all
seven router tests. The four-row variant reached 174.63 ± 0.40 against
174.54 ± 0.40 for the eight-row control. The existing kernels remain in use.

A gate/up Q3_K and down Q4_0 trial in physical layers 0–19, requantized
from the published IQ4_NL model, reached 180.83 ± 0.25 tg128. Its eight-chunk
prefill perplexity of 2.9578 exceeded the branch tolerance of 0.5% over
2.9401. This trial was excluded from the serving configuration. The down
projection has inner dimension 1152, which requires a fallback for Q3_K
blocks of 256. Splitting the projection on short DSpark verification batches
also left plain single-stream decode faster across the three coding chats.

### Calibrated two-bit experts on one RTX 3090

The runs used all three core passes and eight experts, 16K context, one slot,
Q8_0 K/V, F16 controllers, batched messages, specialized mixes, segmented IQ4
projections and GPU backend sampling. The 700-token figures are mean server
decode rates across the three prompts in `perf/bench.py`. Each Q2 checkpoint
was quantized directly from the publisher's Q8_0 GGUF using its 1,770-chunk
importance matrix. The output head and attention weights use IQ4_NL.

| Gate/up and down in layers 0–19 | Layers 20–23 | tg128 | Served T=0 / 0.7 | Prefill / decode PPL |
|---|---|---:|---:|---:|
| Q3_K / IQ4_NL | Q8_0 | 180.04 ± 0.53 | 171.5 / 170.8 | 2.9553 / 4.0754 |
| Q2_K / reference Q2_0 | IQ4_NL | 207.19 ± 0.40 | 200.2 / 199.5 | 3.3624 / — |
| Q2_K / calibrated Q2_0 | IQ4_NL | 206.56 ± 0.55 | 202.0 / 201.0 | 3.0516 / 4.2080 |

The Q3 row used CPU sampling. The final Q2 row additionally enabled
`IK_CTRL_APPLY8=1` on rank-64 F16 controllers. The eight-lane implementation
passed 17 CUDA/CPU cases and preserved all three 200-token greedy generations
on the reference Q2 weights. Its paired short-context tests gave 207.94 and
207.85 tok/s enabled against 206.64 and 205.45 disabled. On the final model,
five-repeat tg128 was 191.81 ± 0.35 at depth 1024 and 176.58 ± 4.77 at depth
3072. Both two-bit models exceed the established quality tolerance.

The original Q2_0 encoder ignored importance weights and used the largest
absolute weight as the scale. Commit `704a7da` fits the scale and four
existing levels against the importance weights. It uses the same 9.25 GiB
layout and decode kernels. To reproduce the final model from the repo root,
set `SOURCE` to the publisher Q8_0 GGUF, `MATRIX` to its `imatrix.gguf`,
and `OUT` to a new file:

```bash
build/bin/llama-quantize --allow-requantize --imatrix "$MATRIX" \
  --output-tensor-type iq4_nl --token-embedding-type q8_0 \
  --tensor-type 'attn_(q|k|v|output|gate|rel|sink_proj|k_conv_proj|v_conv_proj)\.weight=iq4_nl' \
  --tensor-type 'blk\.(1[0-9]|[0-9])\.ffn_gate_up_exps=q2_k' \
  --tensor-type 'blk\.(1[0-9]|[0-9])\.ffn_down_exps=q2_0' \
  --tensor-type 'blk\.2[0-3]\.ffn_.*_exps=iq4_nl' "$SOURCE" "$OUT" iq4_nl 12
```

With the final kernel and identical sampling settings, the previous dense IQ4
model measured 170.6 / 170.3 tok/s. A fresh launch of the calibrated Q2
candidate repeated at 201.2 / 200.2 tok/s. The GGUF SHA-256 was
`bc7f51192fbc35eeb5eaec6a342028b8588691023a2399d97c29eb633a049d07`.
OpenCode 1.2.5 read a local file and returned its exact marker with both
models. At a 2,880-token initial prompt, tool-call and follow-up generation
rates were 144.8 / 143.1 tok/s with IQ4 and 165.7 / 163.8 with calibrated
Q2. The short benchmark prompt and the OpenCode system prompt exercise
different cache depths.

The live 32K endpoint repeated the six short-prompt 700-token generations
at 201.9 / 201.1 tok/s (T=0 / 0.7). All six exceeded 200 tok/s. With the
full OpenCode tool set, the 16,963-token initial prompt took 8.92 s to
prefill. The 78-token tool call decoded at 140.20 tok/s, and the 105-token
follow-up decoded at 140.66 tok/s and returned the exact file marker.

# Phase 5: long-context decode, prefill, speculative verify (2026-11-21)

Start: `c3ab089`. One RTX 3090 (GPU 1), CUDA 12.8, the live flags (`perf/p5env.sh`), Q8_0 K/V, all three core passes.
The 2-bit model is `Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf`; "IQ4_NL dense" is
`Inkling-2-Mini-IQ4_NL-dense-iq4nl.gguf`. Raw outputs are in `perf/p5/`. Nothing here changes the weights or the KV
cache format, and both changes are on by default (`IK_ATTN_MMA=0` and `IK_CTRL_GEMM_MIN=0` restore the old paths).

## Tables

### Decode at depth (`llama-bench -p 0 -n 128 -d 0,4096,16384,32768 -r 3`, tok/s ± sd)

| Model | Build | 0 | 4,096 | 16,384 | 32,768 |
|---|---|---:|---:|---:|---:|
| 2-bit | before (`c3ab089`) | 209.47 ± 0.30 | 168.69 ± 4.41 | 148.92 ± 3.74 | 131.17 ± 3.21 |
| 2-bit | phase 5 | 210.32 ± 0.32 | 194.88 ± 2.36 | 182.46 ± 2.09 | 168.67 ± 1.78 |
| IQ4_NL dense | before (`IK_ATTN_MMA=0`) | 170.24 ± 1.11 | 146.83 ± 3.43 | 132.43 ± 2.85 | 118.64 ± 2.56 |
| IQ4_NL dense | phase 5 | 176.77 ± 0.49 | 166.09 ± 1.75 | 157.28 ± 1.66 | 146.88 ± 1.30 |

The 2-bit model gains 16% at 4K, 23% at 16K and 29% at 32K. It does not reach 200 tok/s at depth: 195 at 4K, 182 at
16K, 169 at 32K. At 1,024 tokens it is 200.7 (was 191.8).

### Prefill (`llama-bench -p 512,4096,16384 -n 0`, tok/s)

| Model | Build | 512 | 4,096 | 16,384 |
|---|---|---:|---:|---:|
| 2-bit | before | 2,294 | 2,184 | 2,040 |
| 2-bit | phase 5 | 3,213 | 3,018 | 2,782 |
| 2-bit | phase 5, `-b 2048 -ub 2048` | 3,247 | 3,563 | 3,260 |
| IQ4_NL dense | before (`IK_CTRL_GEMM_MIN=0`) | 2,353 | 2,255 | 2,119 |
| IQ4_NL dense | phase 5 | 3,453 | 3,230 | 2,972 |

### Served (2-bit model, 64K slot, `--backend-sampling`, one client)

Short prompts are `perf/bench.py` (three 700-token replies per temperature). Long prompts are `perf/longctx.py`: this
repository's `src/llama-*.cpp` cut to the token count, a 500-token reply, temperature 0, prompt cache off. Time to
first token is measured at the client. "Before" is the same binary with both switches off.

| Build | Short T=0 / 0.7 | 16,053-token prompt: TTFT, decode | 32,054-token prompt: TTFT, decode | VRAM after load |
|---|---:|---:|---:|---:|
| before | 201.9 / 200.7 | 8.46 s (1,910 tok/s), 147.9 tok/s | 17.87 s (1,799 tok/s), 131.1 tok/s | 13.9 GB |
| phase 5 | 206.9 / 205.6 | 6.27 s (2,581 tok/s), 179.4 tok/s | 13.47 s (2,389 tok/s), 165.7 tok/s | 13.9 GB |
| phase 5, `-b 2048 -ub 2048` | — | 5.38 s (3,001 tok/s), 172.7 tok/s | 11.62 s (2,772 tok/s), 160.0 tok/s | 19.5 GB |

The larger micro-batch reads the prompt 16% faster but decodes 4% slower afterwards and takes 5.6 GB more VRAM. The
cause of the slower decode was not investigated. The default micro-batch stays the recommendation.

### Quality

| Check | 2-bit before | 2-bit phase 5 | IQ4_NL dense before | IQ4_NL dense phase 5 |
|---|---:|---:|---:|---:|
| Prefill PPL, 8 × 2048 | 3.0516 | 3.0530 | 2.9458 | 2.9471 |
| Decode-path PPL, 2 × 1024, `-ub 1` | 4.2080 | 4.2036 | 4.0990 | 4.0688 |
| Decode-path PPL, 6 × 1024, `-ub 1` | 3.7258 | 3.7195 | 3.6260 | 3.6149 |

- The IQ4_NL two-chunk decode PPL moved by −0.74%, past the 0.5% band in the favourable direction; its standard
  error is 0.29. On six chunks the change is −0.31% (IQ4_NL) and −0.17% (2-bit).
- Greedy probes (three 200-token prompts, 2-bit model): each diverges from the scan-off output at a near-tie. The
  top-two logprob margins at the first differing token are 0.038, 0.030 and 0.028 (`perf/margin_probe.py`).
- The 827-token mixed-prefill probe matches the unfused, unrotated reference for all 12 tokens.
- `test-backend-ops`: `INKLING_ATTN` 16/16, `LOOP_CTRL_DOWN` 14/14, `LOOP_CTRL_APPLY` 23/23 (six new batch cases for each
  controller op, F32 and F16 weights).

### Speculative decoding (DSpark, existing batched verify; 2-bit model, 384-token replies, temperature 0)

`perf/spec_probe.py`, a fresh server per row group, drafter `dspark-Inkling-2-Mini-Q8_0.gguf` on the same GPU.

| Prompt | Prompt tokens | Plain tok/s | DSpark max 3 tok/s | Accepted / proposed | DSpark max 1 tok/s | Accepted / proposed |
|---|---:|---:|---:|---:|---:|---:|
| New code | 47 | 208.8 | 128.2 | 223 / 474 (47%) | 120.6 | 174 / 208 (84%) |
| Edit of given code | 1,025 | 197.3 | 191.2 | 282 / 303 (93%) | 123.9 | 186 / 196 (95%) |
| Prose | 36 | 207.4 | 107.5 | 190 / 575 (33%) | 110.7 | 144 / 239 (60%) |
| Edit at 16K context | 16,044 | 179.2 | 136.6 | 265 / 352 (75%) | 97.8 | 182 / 201 (91%) |

DSpark is slower than plain decode on every prompt, including the edit with 93% acceptance. The grouped-expert
verify was not built in this round (see below), so these rows are the unchanged batched path.

## What was done

### 1. Tensor-core scan over the Q8_0 KV cache (`71929ea` and `9d27343`; tags `p5-mma-attn-d32k-164tok`, `p5-final-d32k-169tok`)

Profile first. Going from 16K to 32K cost 0.91 ms per token for 285 MB of K and V (8 full-attention visits × 2,176
bytes per cell): 313 GB/s, against 891 GB/s measured for plain reads on this card. The old scan was bound by its
arithmetic, not by memory: every lane ran a 7-shuffle reduction per cell and `expf` per cell and head.

The new scan runs inside the fused attention kernel and reads the same Q8_0 cache:

- Steps of 16 cells per warp. `Q·Kᵀ` and `P·V` are `mma.sync m16n8k16` products with FP32 accumulation, 16 per step.
- The int8 quants become FP16 exactly, in registers: two bytes are placed in the bit pattern `0x6400 | n` (the value
  1024 + n) and one `sub.rn.f16x2` gives n − 128.
- The per-block Q8_0 scales stay in FP32. On the K side they multiply each block's partial dot product. On the V
  side each product takes 16 dims of one block (two per lane group), so the scale folds into P.
- Q and P enter as two FP16 limbs (value and rounding remainder) in the spare columns of the 8-column operand, so the
  result equals the FP32 kernel up to summation order. That is why it can be the default.
- Q8_0 rows are 34-byte blocks, so half the quants are not 4-byte aligned. The misaligned blocks are read as a rotated
  stream (dims 2..31, 0..1) and Q uses the same permutation; no unaligned loads.
- One online softmax per lane group, rescaled only when a step raises the maximum.

Result: 16K→32K now costs 0.45 ms per token (about 640 GB/s, of which 0.05 ms is CPU-side mask work). Decode is
169 tok/s at 32K, from 131.

### 2. Loop controllers as GEMMs in batches (`8b0afdd`, tag `p5-prefill-ctrl-gemm-3062tok`)

An nsys trace of a 4,096-token prefill (1,884 ms of kernel time) showed `loop_ctrl_apply` at 21.8% and
`loop_ctrl_down` at 13.2%. Both were written for one token: a block per (rank, token) and a warp per output element.
For 8 or more tokens they are now `D = W_downᵀ·[rmsnorm(a); rmsnorm(msg)]` and `[G, F] = [W_gain, W_forget]·D` through
`cublasSgemm`, followed by one element-wise kernel. The cuBLAS handle's TF32 mode is switched off around the calls so
the sums stay FP32; F16 controller weights are widened into a scratch buffer per call. The controllers went from
658 ms to about 90 ms of that trace. What remains is `mul_mat_q` 57%, flash attention 8.7%, `rel_bias` 6.5%.

## What did not work

- **Issuing the next step's loads before computing the current one.** 250 registers, no change in tok/s.
- **Two blocks per SM for the fused kernel** (`__launch_bounds__(256, 2)`). The compiler spills 20 words; 1.5% slower.
- **A stand-alone scan kernel with 4-warp blocks**, so an SM holds 16 scanning warps. 165 tok/s at 32K with 160
  blocks, 151 with 328. More concurrent warps made it slower.
- **The same with one block per cell range covering all eight kv heads**, so that the 128-byte lines shared by
  neighbouring heads' 136-byte rows are fetched once. 162 tok/s at 32K: no gain, so line sharing is not the limit.
- **A shared-memory tile filled with 16-byte loads in cache order.** Reading the same bytes with 16-byte sequential
  loads and no arithmetic costs 0.39 ms per 16K cells against 0.53 ms for the 4-byte fragment loads, but with the
  two block barriers per step the full kernel was slower (158 tok/s at 32K). The stand-alone kernel stays in the
  tree, off by default (`IK_ATTN_SCAN_MIN=2048` turns it on).
- **Larger splits on the old kernel** (`IK_ATTN_CELLS_PER_SPLIT` 128/256/512): at most +5% at 4K, nothing deeper.
- **DSpark's `--spec-draft-p-min`** (0.6 to 0.9) as a cheap "draft only when it pays" switch: the server then
  proposes no drafts at all and runs at the plain rate.
- **Nsight Compute** could not be used: `ERR_NVGPUCTRPERM`, the GPU counters need admin rights on this machine.

Knock-out measurements on the final scan (2-bit model, per 16K extra cells): without the scan 0.05 ms (the CPU builds
and uploads the mask and distance arrays, O(cells) per token); K/V loads alone 0.53 ms; no loads 0.31 ms. The step is
about 680 executed instructions for 16 cells of one kv head. Making the two-limb selection branch-free was worth
2.8% at 32K, which says the instruction count still matters.

## Not done

- **Grouped-expert speculative verify (workstream 2).** A trace of a DSpark request (160 verify steps, 17 ms of
  kernel time each) splits a 4-token step into: per-token expert matvecs 4.8 ms (no grouping), dense matvecs 4.0 ms
  (the target's head, the drafter and its head), and about 8 ms of unfused per-layer glue across 1,960 launches. A
  plain token is 4.8 ms. With the measured acceptance (2.4 tokens per step on code, 3.8 on the edit) a 4-token
  window has to cost about 9 ms including the drafter before code breaks even. Reading each expert once would save
  roughly 2 ms. The rest needs multi-token versions of every fused decode operator (attention with in-window keys
  and rollback of the convolution state, router, grouped expert matvec, controllers), as Maya's window path has.
  That is the next piece of work; an adaptive on/off rule only helps once the window is cheap.
- **Lower-bit KV (FP8, Q4_0).** After the tensor-core change the scan is bound by per-step arithmetic and load
  pattern more than by bytes, so halving the bytes would not halve the time. It also changes numerics and would have
  shipped as an opt-in variant. Not attempted.
- **Skipping the prefix on visits that do not need it.** All 8 full-attention visits attend over the whole prefix;
  Maya reports that 0.0% of steps could skip V reads on a real 64K text. Not attempted.

## Sources

1. Project Maya `maya-ik2`, <https://github.com/alesha-pro/maya-inkling-mini> (MIT). `docs/INKLING2-MINI.md` for
   the facts used here (8 of 48 visits attend over the full prefix; attention on tensor cores from the cache into
   registers; GEMM prefill; the grouped-expert window and its adaptive rule, which were studied and not built).
   `src/ik2/ik2_kernels.cu`, `attn_decode_mma_kernel`: the scan's structure, the `mma16816` wrapper and the
   integer-to-FP16 bit pattern are adapted from it. Notice in `NOTICE.inkling` and in the header comment of the scan
   in `ggml/src/ggml-cuda/inkling.cu`.
2. Dao et al., "FlashAttention", arXiv:2205.14135, and Dao, "FlashAttention-2", arXiv:2307.08691: tiled attention
   with an online softmax and FP32 accumulators.
3. Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference" (PyTorch blog, 2023): splitting the keys
   of a single query across blocks and merging (max, sum, partial output) at the end. The split merge predates this
   phase; the split counts were retuned.
4. Milakov and Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867.
5. Dekker, "A floating-point technique for extending the available precision", Numer. Math. 18 (1971): the
   value-plus-remainder split behind the two FP16 limbs. Maya uses the same idea for its expert inputs.
6. NVIDIA PTX ISA, `mma.sync.aligned.m16n8k16` fragment layout and `sub.rn.f16x2`.
7. NVIDIA cuBLAS, `cublasSgemm` and `cublasSetMathMode` (TF32 tensor-op math is the handle default in ggml).
8. llama.cpp PR #32218 (DSpark block drafting) and Leviathan et al., "Fast Inference from Transformers via
   Speculative Decoding", arXiv:2211.17192, for the acceptance arithmetic.

## Risks

- The scan assumes the Q8_0 cache layout of this model: 128-dim heads, rows 4-byte aligned, K and V of the same type.
  Anything else falls back to the previous kernel (F16 caches always do).
- P is scaled by 4096 before it becomes FP16. A V block scale above 16 would overflow; V is RMS-normalized
  (|v| ≤ 11.3, scale ≤ 0.09), so there is a factor of 180 of margin.
- Greedy outputs change at near-ties relative to `c3ab089`, as with every earlier change of summation order.
- The controller GEMMs allocate scratch from the CUDA pool per call (28 MB per controller op at 512 tokens).
- Served 16K and 32K rows are single runs.

## Deploy

The live unit runs the snapshot in `~/inkling-live-bin` (build `c3ab089`) and was restored unchanged. To move it to
this build with the same model and flags, run `bash perf/live/switch-to-p5.sh`. It snapshots `build/bin` into
`~/inkling-live-bin-p5`, stops the unit and starts it from the snapshot. `bash perf/live/restore-live.sh` after a
`systemctl --user stop inkling-3090-experiment` goes back.


# Phase 6: speculative verify as a window of rows (2026-11-21)

Goal: make DSpark pay on one RTX 3090 by verifying 1-8 draft tokens in one step whose rows are bit-identical to
single-token decode, as the basalt and ingine engines do on an RTX 5090. Brief: `perf/PHASE6_BRIEF.md`. Raw outputs:
`perf/p6/`. Commit of the numbers: `2359b45` unless a row says otherwise. Everything new is behind `IK_MR=1`
(and `LLAMA_DSPARK_ADAPTIVE=1` for the policy); without them the build behaves as phase 5.

**Outcome in one paragraph.** A verify step is now a "window": every fused single-token operator takes the rows of
the window, and the logits of each row equal single-token decode bit for bit (700 rows checked, 0 differ; before, a
batched step differed from single-token decode in every row by up to 6.9 logits). The glue of a 4-row window went
from about 4.6 ms in 1,637 launches to about 2.7 ms in 728. The drafter's cost per step was cut by about 3 ms. DSpark
now beats plain decode on edits (**235 vs 197 tok/s, 1.19x; 196 vs 179 at 16K context, 1.10x**) and with the on/off
policy costs 9-10% on short new-code and prose replies instead of 25-35%. It does not reach the 1.6-1.9x of the 5090
engines: on Ampere the expert matvecs are bound by the dp4a work per (token, expert) pair, not by weight bytes, so
reading a shared expert once saves nothing, and a 4-row window costs 2.3 single tokens.

## Correction of the roofline

The 216 tok/s ceiling in the earlier sections and the write-up is for the exact IQ4_NL weights (4.12 GB read per
token). The calibrated 2-bit model reads 1,135 MB dense + 1,019 MB experts + 261 MB head = **2.42 GB per token**: 2.7 ms
at 891 GB/s, **about 369 tok/s**. Plain decode at 210 tok/s (4.76 ms) spends about 2.1 ms on latency, not bandwidth.

## Tables

### DSpark against plain decode (`perf/spec_probe.py`, 2-bit model, 384-token replies, temperature 0, fresh server per row group)

`perf/p6/final_spec.txt`. Drafter `dspark-Inkling-2-Mini-Q8_0.gguf`, `--spec-draft-p-min 0`, `--backend-sampling`.
"Tokens per step" = 384 / (384 - accepted). Every reply of every DSpark row is byte-identical to the plain reply
(`perf/p6/final_spec_equal.txt`, 20 of 20).

| Prompt | Prompt tokens | Plain tok/s | Old batched verify, max 3 (this build without `IK_MR`) | Window verify, max 3 (`IK_MR=1`) | Accepted / proposed | Tokens per step | Window + policy (`LLAMA_DSPARK_ADAPTIVE=1`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| New code | 47 | 208.3 | 151.4 | 158.1 (0.76x) | 219 / 488 (45%) | 2.33 | 187.3 (0.90x) |
| Edit of given code | 1,025 | 197.0 | 210.1 | **235.3 (1.19x)** | 282 / 303 (93%) | 3.76 | 236.6 (1.20x) |
| Prose | 36 | 207.1 | 126.4 | 139.2 (0.67x) | 196 / 556 (35%) | 2.04 | 188.3 (0.91x) |
| Edit at 16K context | 16,044 | 178.6 | 155.9 | **195.6 (1.10x)** | 275 / 322 (85%) | 3.52 | 191.2 (1.07x) |

Phase 5 on the same prompts (old drafter, old verify): 128.2 / 191.2 / 107.5 / 136.6 tok/s. The "old batched verify"
column here already has the cheaper drafter of this phase, which is where its gain over phase 5 comes from.

Longer drafts lose: max 5 with the policy 176.5 / 231.6 / 177.3 / 174.5; max 7 without it 93.8 / 207.6 / 77.8 / 161.5
(each extra draft position costs a full-vocabulary matvec in the drafter and a row in the window). The confidence
cutoff (`--spec-draft-p-min` as sigmoid of the top-2 logit margin, `perf/p6/sweep1.txt`) raises acceptance (code: 81%
of proposed at 0.8) but not speed (161 tok/s on code, 232 on the edit): a step that drafts nothing still pays the drafter.
An OpenCode-style agent replay was not built.

### Cost of a window (`perf/p6/mr_check.cpp`, real text after a 300-token prefix, logits of every row read back)

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Window (`IK_MR=1`), ms | 4.80 | 8.92 | 11.02 | 13.13 | 15.70 | 18.32 | 20.68 | 22.88 |
| Old batched path, ms | 4.78 | 11.88 | 13.21 | 15.42 | | 19.21 | | 24.53 |
| Distinct experts per layer visit | 8 | 12.7 | 16.3 | 20.2 | 22.8 | 23.9 | 27.9 | 31.0 |

(With one output row and random tokens, `llama-bench -p N`: 7.3 / 12.2 ms for 2 / 4 rows against 8.45 / 11.8 before;
random tokens share no experts and that bench never took the all-rows head.) A second token brings 4.7 new experts per
visit, a fourth 3.9: the overlap ingine reports (4.57) is there, it just does not buy time on this card.

Where a 4-row window of a DSpark step goes (nsys, edit prompt, per step): expert matvecs 5.2 ms (96 launches, 54 us),
dense matvecs 2.7 ms (target head 0.62, Q|K|V 0.89 + 0.61, output projection 0.57), attention 1.6 ms, router 0.55,
the rest of the glue about 1 ms; drafter about 2 ms (4 full-vocabulary matvecs 1.34 ms); host about 2.3 ms.

### Plain decode regression and quality

| | depth 0 | 4K | 16K | 32K |
|---|---:|---:|---:|---:|
| Before (`d43d1d0`), tok/s | 211.11 ± 0.59 | 195.90 ± 2.22 | 183.87 ± 1.95 | 169.50 ± 1.91 |
| Now (`26d08df`), tok/s | 209.91 ± 0.78 | 194.09 ± 2.31 | 182.57 ± 2.05 | 168.32 ± 1.63 |

The single-token kernels are the same template instances as before (the window variants are separate instances); the
0.6-0.9% difference is within the run-to-run spread of these benches (phase 5 final: 210.32 / 194.88 / 182.46 / 168.67).
Prefill PPL (8x2048) 3.0530 and decode-path PPL (2x1024, `-ub 1`) 4.2036: both identical to phase 5.
`test-backend-ops`: INKLING_ATTN 16/16, INKLING_ROUTER 11/11 (4 window cases), moe_sum_conv 6/6 (4 window cases, listed
under CONCAT), SHORTCONV_RES 28/28, LOOP_CTRL_APPLY 23/23, LOOP_CTRL_DOWN 14/14, RES_MIX 70/70, MUL_MAT 2696/2696,
MUL_MAT_ID 1081/1081. `perf/greedy.sh` was not rerun (the plain path is unchanged and both PPLs are identical).

Window equality (`perf/p6/mr_check.txt`): 280 rows in 56 windows of 2-8 rows, 240 rows in 48 windows with partial
rollbacks after a 300-token prefix, 180 rows in 36 windows after 1,500 tokens (crossing the 1,536-cell boundary of both
caches): 0 rows differ, max logit difference 0. The same tool on the old batched path: every row differs.

## What was done

1. **Window graph** (`src/models/inkling2.cpp`, `graph::mr`): a ubatch of 2-8 tokens of one sequence keeps the direct
   recurrent-state access, the Q8_1 activations and every fused op of single-token decode.
2. **Row-exact dense matvecs** (`mmvq.cu` `ik_mmvq_mr`): up to 8 activation columns with the warps, blocks per thread
   and reduction order of `mul_mat_vec_q<type, 1>`; the weight block of an iteration is read once for all columns.
   `ik_segmented_projection_rows` does the same for the Q|K|V projection. Requested per node with
   `op_params[15] == GGML_IK_MR_MAGIC`.
3. **Experts**: `ik_mmvq_mr_pairs`, a warp per (token, slot) pair that walks the blocks of each warp of the
   single-column kernel in turn and adds the partial sums in its order (bit-identical, no shared memory). The grouped
   kernel (a device-side plan, the owner pair computes every column of its expert, each distinct expert streamed once)
   is kept behind `IK_MR_GROUPED=1`; `IK_MR_STATS=1` prints the distinct-expert counts.
4. **Attention**: `inkling_attn_kv_rows_kernel` writes the K/V of all rows (convolution over the state and the rows so
   far, post-norm, cache store, state rollback planes); `inkling_attn2_kernel<MR>` then runs a block set per row over the
   cache, with the row's own mask and distances. A row that sees nothing past the 256-cell pad it would have had as a
   single token takes the splits of that smaller cache, so its partial sums group identically.
5. **Router, MoE sum + convolution, output convolution, loop controllers** with a row dimension
   (`inkling_router2_kernel<MR>`, `inkling_moe_sum_conv_rows_kernel`, `shortconv_res_rows_f32`, `ik_ctrl_apply8_rows`).
   The state rollback planes follow `build_conv_state_view` (slot k: the state k tokens back).
6. **Drafter** (`src/models/dspark.cpp`, `common/speculative.cpp`): the Markov head stops at `--spec-draft-n-max`
   instead of running the whole block (each position is a full-vocabulary matvec plus a first-maximum search), only
   the read positions' logits are computed and copied, and the first maximum is a scan of the logits instead of a
   sampler chain over a 200K-entry candidate array. The drafts are unchanged (same accepted / proposed counts).
7. **Policy**: `--spec-draft-p-min` works for this drafter as a top-2 margin confidence; `LLAMA_DSPARK_ADAPTIVE=1`
   measures tokens per second of drafted and plain steps online and drafts only while drafting is ahead (32 plain
   steps after a loss, doubling to 256, then a probe).

## What did not work

- **Reading each distinct expert once.** The grouped kernel took 65-73 us per 4-row launch against 51 us for the old
  per-token kernel and 54 us for the per-pair kernel, with 20 distinct experts instead of 32 pairs. Per (expert, column)
  the time is the same as single-token decode (about 1.9 us): the kernel is bound by its dp4a and activation loads per
  column, which sharing the weight bytes does not reduce. More rows per block (8 instead of 2) and carrying the expert
  id in the plan changed nothing.
- **Tensor cores for the experts were not built.** With 2-bit weights unpacked on the fly, an mma step needs about as
  many operations per element to build its operand as dp4a spends, and only 1.6 columns share an expert in a 4-row
  window, so the expected gain is small on this card. ingine and basalt read weights stored for their kernels.
- **The confidence cutoff alone** (see the table notes): shorter windows, same speed.
- The first grouped kernel (8 guarded columns, plan computed in every block) was 2x slower than the old path.

## Not done

- Workstream 4's shortlist or screened drafter head: the drafter still runs `n_max + 1` full-vocabulary IQ4_NL matvecs
  per step (1.34 ms at max 3). Folding `output x hidden_out` into one 201K x rank matrix (basalt) is the next cut.
- The drafter's bookkeeping when no draft is made (target features copied to the host and injected every step) costs
  3% of plain speed (202 vs 209 tok/s with `LLAMA_DSPARK_ADAPTIVE=2`, never draft).
- Workstream 6 (single-token slack), the OpenCode replay, window cases for INKLING_ATTN / MUL_MAT in `test-backend-ops`
  (the end-to-end `mr_check` covers them on the real model), the CPU `shortconv_res` with the state in place.

## Sources

- ingine, <https://github.com/satellitedown/inkling-2-mini-abliterated-ingine> (Apache-2.0): 1-8 token columns per
  decode kernel for the same weight traffic, expert dedup across the window, verifying only confident drafts, the
  bit-lock test of M = 1..8 outputs. Ideas only; no code was copied.
- basalt, <https://github.com/jesdga95/basalt> (MIT): a plan kernel that groups (row, slot) pairs by expert, per-row
  results bit-identical to single-row calls, a window policy learned online from measured cost. Ideas only; no code
  was copied. Neither repository was read line by line in this phase; the descriptions in the brief were used.
- llama.cpp's recurrent-state rollback planes (`build_conv_state_store`, `n_rs_seq`) and its `mul_mat_vec_q` /
  `mul_mat_vec_q_moe` kernels, which the window kernels mirror.

## Risks

- `IK_MR=1` needs the live configuration: IQ4_NL or Q6_K dense projections with `IK_SEGMENTED_*`, Q8_0 or F16 KV, one
  sequence, GA10x. Other models fall back to the old batched path (`graph::mr` false).
- Bit equality with plain decode holds when a row's visible cells and cache size match single-token decode. A row whose
  later window rows raise the cache pad while other used cells sit in that pad would group its sums differently (not
  wrong, not bit-equal); not seen in the checks.
- `LLAMA_DSPARK_HEAD_N` is set by the drafter and makes its graph return `n_max + 1` logit columns; a caller that asks
  the drafter context for more outputs would read past them.
- The policy is tuned on 384-token replies; its probes are a larger share of short replies than of long ones.

## Deploy

The live server is back on the phase 5 binaries (`bash perf/live/restore-p5.sh`), unchanged. Plain decode gains nothing
from this build, so there is no reason to switch for plain serving. DSpark is worth turning on only for edit-heavy use
(rewrites of given code: +19%, +10% at 16K context); it costs about 10% on short new-code and prose replies with the
policy and 0.4 GB of VRAM for the drafter. To try it (same model and port, snapshot of this build):

```bash
systemctl --user stop inkling-3090-experiment
mkdir -p ~/inkling-live-bin-p6 && cp -a build/bin/. ~/inkling-live-bin-p6/
systemd-run --user --unit=inkling-3090-experiment --working-directory=$HOME/llama.cpp-fast \
  --setenv=CUDA_VISIBLE_DEVICES=1 --setenv=LLAMA_INKLING_CTRL_F16=1 --setenv=IK_BATCH_MESSAGES=1 --setenv=IK_SEGMENTED_Q6=1 \
  --setenv=IK_SEGMENTED_IQ4=1 --setenv=IK_CTRL_APPLY8=1 --setenv=IK_MIX_SPECIALIZE=1 --setenv=IK_MR=1 \
  --setenv=LLAMA_DSPARK_ADAPTIVE=1 --setenv=LD_LIBRARY_PATH=$HOME/inkling-live-bin-p6 \
  $HOME/inkling-live-bin-p6/llama-server -m $HOME/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 1 --backend-sampling --host 127.0.0.1 --port 8031 --alias inkling-mini \
  --spec-type draft-dspark --spec-draft-model $HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-draft-n-max 3 --spec-draft-p-min 0
```

This command was not run at 64K context in this phase (the measurements used a 32K slot). Back to phase 5:
`systemctl --user stop inkling-3090-experiment && bash perf/live/restore-p5.sh`.
