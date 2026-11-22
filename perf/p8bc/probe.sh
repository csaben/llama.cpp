#!/usr/bin/env bash
# Usage: probe.sh OUTPUT LABEL WIDTH [additional server flags].
set -eu -o pipefail
: "${SPEC_PORT:?source perf/p8bc/env.sh first}"
[[ $GPU == 0 && $CUDA_VISIBLE_DEVICES == 0 ]]
out=$1
label=$2
width=$3
shift 3
args=(--backend-sampling)
if (( width )); then
    args+=(--spec-type draft-dspark --spec-draft-model "$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf"
           --spec-draft-n-max "$width" --spec-draft-p-min 0)
fi
mkdir -p "$out"
SPEC_OUT="$out" perf/spec_probe.py "$label" -- "${args[@]}" "$@" | tee "$out/$label.txt"
