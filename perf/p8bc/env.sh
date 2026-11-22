# Source from the worktree root.
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID GPU=${P8_GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export MODEL="$Q2" LD_LIBRARY_PATH="$PWD/build/bin"
export IK_MR=1 IK_MMA=3 IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0
export LLAMA_DSPARK_DEV=1 LLAMA_DSPARK_ROUND=1 LLAMA_DSPARK_GRAPHS=1 LLAMA_DSPARK_POLICY=1
export LLAMA_ARG_OVERRIDE_TENSOR='token_embd.weight=CUDA0'
export LLAMA_ARG_SPEC_DRAFT_OVERRIDE_TENSOR='mask_embd.weight=CUDA0'
export SPEC_CTX=65536 SPEC_REQUIRE_IDS=1 SPEC_FIXED_LENGTH=1 SPEC_PORT=18431
export SPEC_KINDS=code,edit,prose,edit@16K,edit@32K,agent
export SPEC_PROMPTS_FILE=perf/p7b/prose-prompts.json
export P8_DRAFT="$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf"
