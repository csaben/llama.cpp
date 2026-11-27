#!/usr/bin/env bash
set -euo pipefail
label=${1:?label}
raw=${B_SCAN_RAW:-perf/p11/b/screen/scan}
work=${B_SCAN_WORK:-build/p11/micro}
mkdir -p "$raw" "$work"
exec {label_fd}>"$work/$label-sweep.lock"
flock "$label_fd"
args=()
for length in ${B_SCAN_LENGTHS:-4096 16384 65536}; do
    for rows in ${B_SCAN_WIDTHS:-1 2 3 4 5 6 7}; do
        name="$label-$length-$rows"
        if [[ ! -s "$raw/$name.tsv" ]]; then args+=("$length" "$rows" "$work/$name.attn"); fi
    done
done
if ((${#args[@]} == 0)); then exit 0; fi
base="$work/$label-sweep"
B_OPERATOR_SWEEP=1 B_MIN_FREE_MIB=1000 perf/p11/b/gpu.sh env LD_LIBRARY_PATH="${B_LIBS:-$HOME/inkling-live-bin-p9-72fd097}" IK_MMA=3 IK_MR=1 \
    IK_WINDOW_COLS="${IK_WINDOW_COLS:-1}" GGML_CUDA_GRAPH_CHUNK_SIZE=0 build/p11/sweep/attention --cases "${args[@]}" >"$base.times" 2>"$base.log"
for ((i=0; i<${#args[@]}; i+=3)); do
    length=${args[i]} rows=${args[i+1]} path=${args[i+2]}
    times=$(awk -v d="$length" -v n="$rows" '$1==d && $2==n && NF==11 {print; found++} END {exit found!=1}' "$base.times")
    test -s "$path"
    digest=$(sha256sum "$path" | cut -d' ' -f1)
    printf '%s\t%s\n' "$times" "$digest" >"$raw/$label-$length-$rows.tsv"
    rm "$path"
done
