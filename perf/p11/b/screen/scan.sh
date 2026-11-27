#!/usr/bin/env bash
set -euo pipefail
label=${1:?label}
raw=${B_SCAN_RAW:-perf/p11/b/screen/scan}
work=${B_SCAN_WORK:-build/p11/micro}
mkdir -p "$raw" "$work"
for length in ${B_SCAN_LENGTHS:-4096 16384 65536}; do
    args=()
    widths=()
    for rows in ${B_SCAN_WIDTHS:-1 2 3 4 5 6 7}; do
        name="$label-$length-$rows"
        if [[ -s "$raw/$name.tsv" ]]; then continue; fi
        args+=("$rows" "$work/$name.attn")
        widths+=("$rows")
    done
    if ((${#widths[@]} == 0)); then continue; fi
    base="$work/$label-$length"
    B_OPERATOR_SWEEP=1 B_MIN_FREE_MIB=1000 perf/p11/b/gpu.sh env LD_LIBRARY_PATH="${B_LIBS:-$HOME/inkling-live-bin-p9-72fd097}" IK_MMA=3 IK_MR=1 \
        IK_WINDOW_COLS="${IK_WINDOW_COLS:-1}" GGML_CUDA_GRAPH_CHUNK_SIZE=0 build/p11/sweep/attention "$length" "${args[@]}" >"$base.times" 2>"$base.log"
    for rows in "${widths[@]}"; do
        name="$label-$length-$rows"
        times=$(awk -v d="$length" -v n="$rows" '$1==d && $2==n && NF==11 {print; found++} END {exit found!=1}' "$base.times")
        test -s "$work/$name.attn"
        digest=$(sha256sum "$work/$name.attn" | cut -d' ' -f1)
        printf '%s\t%s\n' "$times" "$digest" >"$raw/$name.tsv"
        rm "$work/$name.attn"
    done
done
