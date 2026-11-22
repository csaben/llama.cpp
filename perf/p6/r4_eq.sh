#!/usr/bin/env bash
# Reproduce the original 16K-slot edit case. Use r4_spec.sh for both measurement tables.
set -euo pipefail
cd "$(dirname "$0")/../.."
source perf/p5env.sh
export EQ=${EQ:-perf/p6/r4_eq} MAXT=${MAXT:-3000} KINDS=${KINDS:-edit} IK_MR=1 IK_MMA=3
mkdir -p "$EQ"
label=${1:?usage: r4_eq.sh LABEL plain|N [extra server args]}
width=${2:?missing verify width}
shift 2
args=()
if [[ $width != plain ]]; then
    draft=${DRAFT_FILE:-$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf}
    args+=(--spec-type draft-dspark --spec-draft-model "$draft" --spec-draft-n-max "$width" --spec-draft-p-min 0)
fi
if [[ ${POLICY:-0} == 1 ]]; then export LLAMA_DSPARK_ADAPTIVE=1; fi
exec python3 perf/p6/r4_repro/eq.py "$label" "${args[@]}" "$@"
