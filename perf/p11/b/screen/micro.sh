#!/usr/bin/env bash
# One archived prototype, p9 or near-neighbor control per leg, run on one actual expert tensor.
set -euo pipefail
runtime=${1:?runtime} suite=${2:?tile/verify} label=${3:?leg}
model="$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf"
raw=perf/p11/b/screen/micro
work=build/p11/micro
mkdir -p "$raw" "$work"
widths=(512)
if [[ "$suite" == verify ]]; then widths=(2 3 4 5 6 7); fi
for projection in gate down; do
    args=()
    sizes=()
    for cols in "${widths[@]}"; do
        name="$suite-$label-$projection-$cols"
        if [[ -s "$raw/$name.tsv" ]]; then continue; fi
        args+=("$cols" "$work/$name.f32")
        sizes+=("$cols")
    done
    if ((${#sizes[@]} == 0)); then continue; fi
    base="$work/$suite-$label-$projection"
    B_OPERATOR_SWEEP=1 B_MIN_FREE_MIB=1000 perf/p11/b/gpu.sh env LD_LIBRARY_PATH="$runtime" IK_MMA="${B_MMA:-3}" IK_MR=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0 \
        build/p11/sweep/expert "$model" "$projection" "${args[@]}" >"$base.times" 2>"$base.log"
    for cols in "${sizes[@]}"; do
        name="$suite-$label-$projection-$cols"
        test "$(stat -c %s "$work/$name.f32")" -eq "$((4*2304*8*cols))"
        times=$(awk -v p="$projection" -v n="$cols" '$1==p && $2==n && NF==12 {print; found++} END {exit found!=1}' "$base.times")
        digest=$(sha256sum "$work/$name.f32" | cut -d' ' -f1)
        printf '%s\t%s\n' "$times" "$digest" >"$raw/$name.tsv"
        rm "$work/$name.f32"
    done
done
