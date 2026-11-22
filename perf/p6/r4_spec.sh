#!/usr/bin/env bash
# Run fresh GPU1 servers. The long table suppresses EOS so each prompt exercises 3000 decode tokens.
set -euo pipefail
cd "$(dirname "$0")/../.."
source perf/p5env.sh
export MODEL=$Q2 LD_LIBRARY_PATH="$PWD/build/bin"
export SPEC_MAXT=${SPEC_MAXT:-3000} SPEC_REPS=${SPEC_REPS:-1}
export SPEC_OUT=${1:?usage: r4_spec.sh OUTPUT_DIRECTORY [ROWS...]}
shift
if (( $# == 0 )); then set -- plain plain-mr n2 n3 n4 n2-adapt n3-adapt n4-adapt; fi
mkdir -p "$SPEC_OUT"
for label in "$@"; do
    args=(--backend-sampling)
    case "$label" in
        plain) flags=() ;;
        plain-mr) flags=(IK_MR=1 IK_MMA=3) ;;
        n[234]|n[234]-adapt)
            args+=(--spec-type draft-dspark --spec-draft-model "$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf"
                --spec-draft-n-max "${label:1:1}" --spec-draft-p-min 0)
            flags=(IK_MR=1 IK_MMA=3)
            if [[ $label == *-adapt ]]; then flags+=(LLAMA_DSPARK_ADAPTIVE=1); fi ;;
        *) echo "unknown row $label" >&2; exit 2 ;;
    esac
    env -u IK_MR -u IK_MMA -u LLAMA_DSPARK_ADAPTIVE "${flags[@]}" SPEC_FIXED_LENGTH=1 perf/spec_probe.py "$label" -- "${args[@]}" | tee "$SPEC_OUT/$label.txt"
done
