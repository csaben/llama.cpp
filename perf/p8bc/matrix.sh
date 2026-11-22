#!/usr/bin/env bash
# Run the Phase 7c request matrices and compare every message and token ID.
set -eu -o pipefail
cd "$(dirname "$0")/../.."
. perf/p8bc/env.sh
if [[ -n ${P8_LIBS:-} ]]; then export LD_LIBRARY_PATH="$P8_LIBS:$LD_LIBRARY_PATH"; fi
out=${1:?usage: matrix.sh OUTPUT_DIRECTORY}
for budget in 384 3000; do
    export SPEC_MAXT=$budget
    for suite in main prose fixed; do
        dir="$out/$suite/$budget"
        export SPEC_KINDS=code,edit,prose,edit@16K,edit@32K,agent
        rows=(n2 n3 n4 n2-adapt n3-adapt n4-adapt n6-adapt)
        export SPEC_REPS=3 LLAMA_DSPARK_POLICY=1
        case $suite in
            fixed) rows=(n2 n3 n4); export LLAMA_DSPARK_POLICY=0 ;;
            prose) rows=(n2-adapt n3-adapt n4-adapt n6-adapt); export SPEC_REPS=5 SPEC_KINDS=letter,science,scene,review,dialogue ;;
        esac
        if [[ $budget == 3000 ]]; then export SPEC_REPS=1; fi
        unset LLAMA_DSPARK_ADAPTIVE
        bash perf/p8bc/probe.sh "$dir" plain 0
        for row in "${rows[@]}"; do
            unset LLAMA_DSPARK_ADAPTIVE
            if [[ $row == *-adapt ]]; then export LLAMA_DSPARK_ADAPTIVE=1; fi
            bash perf/p8bc/probe.sh "$dir" "$row" "${row:1:1}"
            python3 perf/p6/r4_cmp.py "$dir" plain "$row" | tee "$dir/$row.equal"
        done
    done
done
