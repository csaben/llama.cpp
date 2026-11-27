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
for cols in "${widths[@]}"; do
    for projection in gate down; do
        name="$suite-$label-$projection-$cols"
        if [[ -s "$raw/$name.tsv" ]]; then continue; fi
        B_MIN_FREE_MIB=1000 perf/p11/b/gpu.sh env LD_LIBRARY_PATH="$runtime" IK_MMA=3 IK_MR=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0 \
            build/p11/expert "$model" "$projection" "$cols" "$work/$name.f32" >"$work/$name.times" 2>"$work/$name.log"
        test "$(stat -c %s "$work/$name.f32")" = "$((4*2304*8*cols))"
        digest=$(sha256sum "$work/$name.f32" | cut -d' ' -f1)
        printf '%s\t%s\n' "$(cat "$work/$name.times")" "$digest" >"$raw/$name.tsv"
        rm "$work/$name.f32"
    done
done
