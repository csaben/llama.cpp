#!/usr/bin/env bash
set -euo pipefail
if [[ -z ${B_LOCAL_GROUP_FD:-} ]]; then exec perf/p11/b/screen/local-group.sh "$0" "$@"; fi
label=${1:?label}
raw=perf/p11/b/screen/scan
work=build/p11/micro
mkdir -p "$raw" "$work"
for length in ${B_SCAN_LENGTHS:-4096 16384 65536}; do
    for rows in ${B_SCAN_WIDTHS:-1 2 3 4 5 6 7}; do
        name="$label-$length-$rows"
        if [[ -s "$raw/$name.tsv" ]]; then continue; fi
        B_MIN_FREE_MIB=1000 perf/p11/b/gpu.sh env LD_LIBRARY_PATH="${B_LIBS:-$HOME/inkling-live-bin-p9-72fd097}" IK_MMA=3 IK_MR=1 \
            IK_WINDOW_COLS="${IK_WINDOW_COLS:-1}" GGML_CUDA_GRAPH_CHUNK_SIZE=0 build/p11/attention "$length" "$rows" "$work/$name.attn" >"$work/$name.times" 2>"$work/$name.log"
        digest=$(sha256sum "$work/$name.attn" | cut -d' ' -f1)
        printf '%s\t%s\n' "$(cat "$work/$name.times")" "$digest" >"$raw/$name.tsv"
        rm "$work/$name.attn"
    done
done
