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

## Baseline decode graph (per token, 1 stream)
7764 non-view ggml nodes/token -> ~6100 kernel launches + ~1070 matvecs. Op histogram:
MUL_MAT 977, ADD 872, MUL 858, SCALE 629, CONCAT 548, RMS_NORM 514, CONT 480, CPY 337, TANH 289, FILL 240,
SUM_ROWS 192, DIV 192, SSM_CONV 192, GET_ROWS 146, CLAMP 145, SQR 144, SQRT 144, SUB 144, SIGMOID 144,
REPEAT 120, SOFT_MAX 120, SET_ROWS 96, MUL_MAT_ID 96, PAD 48, FLASH_ATTN 48, ARGSORT 48.
