#!/usr/bin/env bash
set -euo pipefail
source perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
label=${1:?runtime label}
export LD_LIBRARY_PATH="${2:?runtime library directory}:$PWD/build/bin"
out=perf/p8m/bench
mkdir -p "$out"
args=(-m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 3 -o json)
for mode in plain mr; do
    flags=()
    if [[ $mode == mr ]]; then flags=(IK_MR=1 IK_MMA=3 IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0); fi
    env "${flags[@]}" build/bin/llama-bench "${args[@]}" -p 0 -n 128 -d 0,4096,16384,32768 >"$out/$label-$mode-decode.json" 2>"build/p8m/$label-$mode-decode.log"
done
IK_MR=1 IK_MMA=3 IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0 \
    build/bin/llama-bench "${args[@]}" -p 4096,16384 -n 0 -b 2048 -ub 512 >"$out/$label-prefill.json" 2>"build/p8m/$label-prefill.log"
