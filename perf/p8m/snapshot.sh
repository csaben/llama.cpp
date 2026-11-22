#!/usr/bin/env bash
# Probe the staged server and its libraries on a separate localhost port.
set -euo pipefail
export P8_GPU=1
source perf/p8bc/env.sh
snapshot=${1:?snapshot directory}
width=${2:?release width}
export LD_LIBRARY_PATH="$snapshot" SPEC_SERVER_BIN="$snapshot/llama-server"
export SPEC_PORT=18433 SPEC_KINDS=edit,prose,agent
export SPEC_MAXT=384 SPEC_REPS=5 LLAMA_ARG_UBATCH=512 LLAMA_ARG_SPEC_DRAFT_N_UBATCH=512
out=perf/p8m/snapshot-probes
for label in plain-before release model-only plain-after; do
    n=0
    export LLAMA_DSPARK_POLICY=1 LLAMA_DSPARK_ADAPTIVE=1
    if [[ $label == release || $label == model-only ]]; then n=$width; fi
    if [[ $label == model-only ]]; then export LLAMA_DSPARK_POLICY=0; fi
    LLAMA_DSPARK_ADAPTIVE_LOG=1 SPEC_SERVER_LOG="build/p8m/snapshot-$label.log" bash perf/p8bc/probe.sh "$out" "$label" "$n" -b 2048
done
python3 perf/p6/r4_cmp.py "$out" plain-before release model-only plain-after >"$out/compare.txt"
for k in ${SPEC_KINDS//,/ }; do
    for ext in ids json; do cmp "$out/plain-before.$k.$ext" "perf/p8m/matrix/main/384/plain.$k.$ext"; done
done
rg 'dspark history: [1-9][0-9]* windows' build/p8m/snapshot-release.log >"$out/history.txt"
