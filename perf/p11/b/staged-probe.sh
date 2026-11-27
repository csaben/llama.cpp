#!/usr/bin/env bash
# Check the staged server on port 18550 with the deployment's vision projector.
set -euo pipefail
export B_LIBS=${1:?snapshot directory} B_MMA=${2:?MMA setting} B_WINDOW_COLS=${3:?window setting} LLAMA_DSPARK_FULL_RATE=${4:?policy setting}
d=${5:?output directory}
mkdir -p "$d"
mmproj="$HOME/models/inkling-2-mini/mmproj-Inkling-2-Mini-Q8_0.gguf"
kinds=code,edit,prose,agent
for leg in plain-before draft1 draft2 plain-after; do
    width=0
    if [[ $leg == draft* ]]; then width=6; fi
    perf/p11/b/gpu.sh perf/p11/b/run.sh probe "$d" "$leg" "$width" "$kinds" 3 384 --mmproj "$mmproj"
    python3 perf/p11/b/probe_complete.py "$d" "$leg" "$kinds" 3 384
    for prompt in code edit prose agent; do
        for ext in json ids; do
            cmp "$d/$leg.$prompt.$ext" "$d/plain-before.$prompt.$ext"
            cmp "$d/$leg.$prompt.$ext" "build/p11/matrix-p9/main/384/plain.$prompt.$ext"
        done
    done
done
echo '24 staged drafted replies and 9,216 IDs matched both staged plain controls and frozen p9.' >"$d/audit.txt"
