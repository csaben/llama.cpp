#!/usr/bin/env bash
# Phase 9's six suites with the Phase 11 model, slot, port and placement.
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=${1:?output directory} placement=${2:?placement}
for pass in controls drafted; do
    for budget in 384 3000; do
        if [[ -n ${P11_MATRIX_BUDGET:-} && $budget != "$P11_MATRIX_BUDGET" ]]; then continue; fi
        for suite in main prose fixed; do
            if [[ -n ${P11_MATRIX_SUITE:-} && $suite != "$P11_MATRIX_SUITE" ]]; then continue; fi
            out="$root/$suite/$budget"
            log="build/p11a/matrix/$(basename "$root")/$suite/$budget"
            mkdir -p "$out" "$log"
            export P11_KINDS=code,edit,prose,edit@16K,edit@32K,agent
            export SPEC_MAXT=$budget SPEC_REPS=3 P11_POLICY=1
            settings=(n2 n3 n4 n2-adapt n3-adapt n4-adapt n6-adapt)
            case $suite in
                prose) settings=(n2-adapt n3-adapt n4-adapt n6-adapt); export SPEC_REPS=5 P11_KINDS=letter,science,scene,review,dialogue;;
                fixed) settings=(n2 n3 n4); export P11_POLICY=0;;
            esac
            if [[ $budget == 3000 ]]; then export SPEC_REPS=1; fi
            if [[ $pass == controls ]]; then settings=(plain); fi
            for setting in "${settings[@]}"; do
                if [[ -n ${P11_MATRIX_SETTING:-} && $setting != "$P11_MATRIX_SETTING" ]]; then continue; fi
                [[ ! -e perf/STOP ]] || exit 75
                mode=draft
                export P11_WIDTH=${setting:1:1} P11_ADAPT=0
                [[ $setting != *-adapt ]] || export P11_ADAPT=1
                [[ $setting != plain ]] || mode=plain
                export P11_OUT="$out" P11_LOG="$log/$setting.server"
                (
                    exec {cell_lock}>"$log/$setting.lock"
                    flock "$cell_lock"
                    if [[ ! -f "$log/$setting.done" ]]; then
                        perf/p11/a/probe.sh "$mode" "$placement" "$setting"
                        if [[ $mode == draft ]]; then
                            SPEC_KINDS="$P11_KINDS" SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$out" plain "$setting" >"$out/$setting.equal"
                        fi
                        touch "$log/$setting.done"
                    fi
                )
                if [[ ${P11_YIELD:-0} != 0 ]]; then sleep "$P11_YIELD"; fi
            done
        done
    done
done
if [[ -z ${P11_MATRIX_BUDGET:-} && -z ${P11_MATRIX_SUITE:-} && -z ${P11_MATRIX_SETTING:-} ]]; then
    python3 perf/p8bc/summarize.py "$root" >"$root/cells.tsv"
fi
