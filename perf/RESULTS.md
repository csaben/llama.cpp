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

## Baseline decode graph (per token, 1 stream)
7764 non-view ggml nodes/token -> ~6100 kernel launches + ~1070 matvecs. Op histogram:
MUL_MAT 977, ADD 872, MUL 858, SCALE 629, CONCAT 548, RMS_NORM 514, CONT 480, CPY 337, TANH 289, FILL 240,
SUM_ROWS 192, DIV 192, SSM_CONV 192, GET_ROWS 146, CLAMP 145, SQR 144, SQRT 144, SUB 144, SIGMOID 144,
REPEAT 120, SOFT_MAX 120, SET_ROWS 96, MUL_MAT_ID 96, PAD 48, FLASH_ATTN 48, ARGSORT 48.
