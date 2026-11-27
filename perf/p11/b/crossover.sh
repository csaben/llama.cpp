#!/usr/bin/env bash
# BRACKET_NEW is the exact runtime directory being compared with frozen p9.
set -euo pipefail
stage=${1:?synthetic or served}
out=${2:?output directory}
mkdir -p "$out"
gpu=(perf/p11/b/gpu.sh perf/p11/b/run.sh)
old="$HOME/inkling-live-bin-p9-72fd097"
new=${BRACKET_NEW:?candidate runtime directory}
for test in ${B_TESTS:-$(if [[ $stage == synthetic ]]; then echo tg0 tg4096 tg16384 tg32768 tg65536 pp4096 pp16384 pp65536; else echo short384 short3000 deep384 deep3000; fi)}; do
    for leg in old1 new1 new2 old2; do
        export B_LIBS=$old B_UBATCH=512 B_WINDOW_COLS=1 B_MMA=3
        if [[ $leg == new* ]]; then
            B_LIBS=$new
            B_UBATCH=${BRACKET_NEW_UB:-512}
            B_WINDOW_COLS=${BRACKET_NEW_WINDOW:-1}
            B_MMA=${BRACKET_NEW_MMA:-3}
        fi
        if [[ ${B_RESUME:-0} == 1 && $test != short* && $test != deep* ]]; then
            if python3 -c 'import json,sys; d,=json.load(open(sys.argv[1])); assert d["avg_ts"] > 0 and len(d["samples_ts"]) == 3' "$out/$test-$leg.json" 2>/dev/null; then continue; fi
        fi
        case $test in
            tg*) "${gpu[@]}" bench -p 0 -n 128 -d "${test#tg}" -r 3 >"$out/$test-$leg.json" 2>"$out/$test-$leg.log" ;;
            pp*) "${gpu[@]}" bench -p "${test#pp}" -n 0 -r 3 >"$out/$test-$leg.json" 2>"$out/$test-$leg.log" ;;
            short*|deep*)
                budget=${test//[a-z]/}
                kinds=code,edit,prose,agent
                if [[ $test == deep* ]]; then kinds=edit@16K,edit@32K; fi
                reps=1
                if [[ $budget == 384 ]]; then reps=3; fi
                cap=6
                if [[ $leg == new* ]]; then cap=${BRACKET_CAP:-6}; fi
                if [[ ${B_RESUME:-0} != 1 ]] || ! python3 perf/p11/b/probe_complete.py "$out/$test" "$leg" "$kinds" "$reps" "$budget"; then
                    "${gpu[@]}" probe "$out/$test" "$leg" "$cap" "$kinds" "$reps" "$budget"
                fi
                if [[ $leg != old1 ]]; then
                    SPEC_KINDS="$kinds" SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$out/$test" old1 "$leg" >"$out/$test/$leg.equal"
                fi ;;
            *) exit 2 ;;
        esac
    done
done
