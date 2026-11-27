#!/usr/bin/env bash
# Run the p9 release matrix with one lock acquisition per server.
set -euo pipefail
out=${1:?output directory}
worker=$BASHPID
for budget in ${B_MATRIX_BUDGETS:-384 3000}; do
    for suite in ${B_MATRIX_SUITES:-main prose fixed}; do
        dir="$out/$suite/$budget"
        kinds=code,edit,prose,edit@16K,edit@32K,agent
        rows=(plain n2 n3 n4 n2-adapt n3-adapt n4-adapt n6-adapt)
        reps=3
        export B_POLICY=1
        case $suite in
            fixed) rows=(plain n2 n3 n4); export B_POLICY=0 ;;
            prose) rows=(plain n2-adapt n3-adapt n4-adapt n6-adapt); reps=5; kinds=letter,science,scene,review,dialogue ;;
        esac
        if [[ $budget == 3000 ]]; then reps=1; fi
        for row in "${rows[@]}"; do
            if [[ -n ${B_MATRIX_ROW:-} && $row != "$B_MATRIX_ROW" ]]; then continue; fi
            width=${row:1:1}
            export B_ADAPT=0
            if [[ $row == plain ]]; then width=0; fi
            if [[ $row == *-adapt ]]; then export B_ADAPT=1; fi
            if ! python3 perf/p11/b/probe_complete.py "$dir" "$row" "$kinds" "$reps" "$budget"; then
                perf/p11/b/gpu.sh perf/p11/b/run.sh probe "$dir" "$row" "$width" "$kinds" "$reps" "$budget"
            fi
            if [[ $row != plain ]]; then
                SPEC_KINDS="$kinds" SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$dir" plain "$row" >"$dir/$row.equal.$worker"
                mv "$dir/$row.equal.$worker" "$dir/$row.equal"
            fi
        done
    done
done
if [[ -z ${B_MATRIX_SUITES:-} && -z ${B_MATRIX_BUDGETS:-} ]]; then
    python3 perf/p8bc/summarize.py "$out" >"$out/cells.tsv.$worker"
    mv "$out/cells.tsv.$worker" "$out/cells.tsv"
fi
