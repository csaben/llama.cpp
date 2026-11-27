#!/usr/bin/env bash
# Run one native llama-bench sweep per lock acquisition.
set -euo pipefail
kind=${1:?tg or pp}
mode=${2:?timed or output}
work=build/p11/bench-grid
mkdir -p "$work"
case $kind in
    tg) args=(-p 0 -n 128 -d 0,4096,16384,32768,65536) ;;
    pp) args=(-p 4096,16384,65536 -n 0) ;;
    *) exit 2 ;;
esac
legs=(old1 new1 new2 old2)
if [[ $mode == output ]]; then
    legs=(old new)
    export B_BENCH_TOOL="$PWD/build/p11/bench-capture/llama-bench"
else
    [[ $mode == timed ]]
fi
for leg in "${legs[@]}"; do
    export B_LIBS="$HOME/inkling-live-bin-p9-72fd097" B_MMA=3 B_WINDOW_COLS=1
    if [[ $leg == new* ]]; then
        B_LIBS="$PWD/build/p11/narrow-bin" B_MMA=7 B_WINDOW_COLS=2
    fi
    name="$work/$kind-$mode-$leg"
    if [[ $mode == output ]]; then export P11_BENCH_DUMP="$PWD/$name.bin"; fi
    if [[ ! -s $name.done ]]; then
        perf/p11/b/gpu.sh perf/p11/b/run.sh bench "${args[@]}" -r 3 >"$name.json" 2>"$name.log"
        python3 - "$name.json" "$kind" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
expected=[(0,128,d) for d in (0,4096,16384,32768,65536)] if sys.argv[2]=='tg' else [(p,0,0) for p in (4096,16384,65536)]
assert [(r['n_prompt'],r['n_gen'],r['n_depth']) for r in rows]==expected
assert all(r['avg_ts']>0 and len(r['samples_ts'])==3 for r in rows)
PY
        date --iso-8601=seconds >"$name.done"
    fi
done
