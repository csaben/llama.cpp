#!/bin/bash
# Round 3 measurement table: plain vs DSpark (pick + shortlist drafter) per prompt, median of SPEC_REPS (default 3) requests.
# usage: perf/p6/r3_spec.sh <out prefix> [rows...]
# rows: plain plain-mr n2 n3 n4 n2-adapt n3-adapt n4-adapt n3-full (full-vocabulary drafter head) n3-r2 (round 2 drafter path)
cd "$(dirname "$0")/../.." && . perf/p5env.sh
OUT=$1; shift; ROWS=${@:-plain plain-mr n2 n3 n4 n2-adapt n3-adapt n4-adapt n3-full}
export MODEL=$Q2 SPEC_OUT=$OUT SPEC_REPS=${SPEC_REPS:-3}
SHORT=$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf
ds() { echo --backend-sampling --spec-type draft-dspark --spec-draft-model "$1" --spec-draft-n-max "$2" --spec-draft-p-min 0; }
for r in $ROWS; do
  case $r in
    plain)         perf/spec_probe.py plain -- --backend-sampling ;;
    plain-mr)      IK_MMA=3 IK_MR=1 perf/spec_probe.py plain-mr -- --backend-sampling ;;
    n[2-8])        IK_MMA=3 IK_MR=1 perf/spec_probe.py $r -- $(ds $SHORT ${r#n}) ;;
    n[2-8]-adapt)  n=${r%-adapt}; IK_MMA=3 IK_MR=1 LLAMA_DSPARK_ADAPTIVE=1 perf/spec_probe.py $r -- $(ds $SHORT ${n#n}) ;;
    n3-full)       IK_MMA=3 IK_MR=1 LLAMA_DSPARK_SHORT=0 perf/spec_probe.py $r -- $(ds $SHORT 3) ;;
    n3-r2)         IK_MMA=3 IK_MR=1 LLAMA_DSPARK_PICK=0 perf/spec_probe.py $r -- $(ds $DRAFT_FOLD 3) ;;
  esac
done 2>&1 | tee -a $OUT.txt
