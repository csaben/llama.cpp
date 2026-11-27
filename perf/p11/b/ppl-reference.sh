#!/usr/bin/env bash
set -euo pipefail
d=${1:?output directory}
mkdir -p "$d"
export B_MODEL="$HOME/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf"
for mode in prefill decode-default decode-mr; do
    opts=(-c 2048 -b 2048 -ub 512 --chunks 8)
    export B_MR=1 B_MMA=3
    if [[ $mode == decode* ]]; then opts=(-c 1024 -b 1024 -ub 1 --chunks 2); fi
    if [[ $mode == decode-default ]]; then B_MR=0 B_MMA=0; fi
    perf/p11/b/gpu.sh perf/p11/b/run.sh ppl "${opts[@]}" >"$d/$mode.log" 2>&1
    rg 'Final estimate:' "$d/$mode.log" >"$d/$mode.txt"
done
