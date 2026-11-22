#!/bin/bash
# Round 4: one eq.py run. usage: [env ...] perf/p6/r4_eq.sh <label> plain|<n_max> [extra server args]
# env: EQ (output dir, default perf/p6/r4_eq), MAXT (3000), KINDS (edit), POLICY=1 -> LLAMA_DSPARK_ADAPTIVE=1
cd "$(dirname "$0")/../.." && . perf/p5env.sh
[ -e perf/STOP ] && { echo "perf/STOP exists"; exit 3; }
export EQ=${EQ:-perf/p6/r4_eq} MAXT=${MAXT:-3000} KINDS=${KINDS:-edit} IK_MR=1 IK_MMA=3
mkdir -p $EQ
L=$1; N=$2; shift 2
DF=${DRAFT_FILE:-$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf}
[ "${POLICY:-0}" = 1 ] && export LLAMA_DSPARK_ADAPTIVE=1
if [ "$N" = plain ]; then python3 perf/p6/r4_repro/eq.py $L "$@"
else python3 perf/p6/r4_repro/eq.py $L --spec-type draft-dspark --spec-draft-model $DF --spec-draft-n-max $N --spec-draft-p-min 0 "$@"; fi
