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
