#!/bin/bash
# Round 2 measurement table: plain vs DSpark (window verify) per prompt. usage: perf/p6/r2_spec.sh <out prefix> [rows...]
# rows: plain plain-mma n2 n3 n4 n3-adapt n2-adapt n4-adapt n3-nofold n3-r1 (default: all but n3-r1)
cd "$(dirname "$0")/../.." && . perf/p5env.sh
OUT=$1; shift; ROWS=${@:-plain plain-mma n2 n3 n4 n2-adapt n3-adapt n4-adapt n3-nofold}
export MODEL=$Q2 SPEC_OUT=$OUT
ds() { echo --backend-sampling --spec-type draft-dspark --spec-draft-model "$1" --spec-draft-n-max "$2" --spec-draft-p-min 0; }
for r in $ROWS; do
  case $r in
    plain)      perf/spec_probe.py plain -- --backend-sampling ;;
    plain-mma)  IK_MMA=3 perf/spec_probe.py plain-mma -- --backend-sampling ;;
    n[2-8])        IK_MMA=3 IK_MR=1 perf/spec_probe.py $r -- $(ds $DRAFT_FOLD ${r#n}) ;;
    n[2-8]-adapt)  n=${r%-adapt}; IK_MMA=3 IK_MR=1 LLAMA_DSPARK_ADAPTIVE=1 perf/spec_probe.py $r -- $(ds $DRAFT_FOLD ${n#n}) ;;
    n3-nofold)  IK_MMA=3 IK_MR=1 perf/spec_probe.py $r -- $(ds $DRAFT 3) ;;
    n3-r1)      IK_MR=1 perf/spec_probe.py $r -- $(ds $DRAFT 3) ;;
  esac
done 2>&1 | tee -a $OUT.txt
