#!/usr/bin/env bash
# Keep the graph cache and adaptive policy enabled in both runtimes.
set -euo pipefail
export P8_GPU=1
source perf/p8bc/env.sh
width=${1:?candidate width}
export SPEC_KINDS=code,edit,prose,agent LLAMA_DSPARK_ADAPTIVE=1
export SPEC_SERVER_BIN="$HOME/inkling-live-bin-p8m-$(git rev-parse --short p8m-merged)/llama-server"
snapshot=$(dirname "$SPEC_SERVER_BIN")
for pass in off1 on1 on2 off2; do
    export LD_LIBRARY_PATH="$snapshot"
    if [[ $pass == off* ]]; then export LD_LIBRARY_PATH="$PWD/build/p8m/lookup-off:$snapshot"; fi
    ldd "$SPEC_SERVER_BIN" | rg 'libllama-common.so.1 =>' >"build/p8m/lookup-$pass-lib.txt"
    for budget in 384 3000; do
        export SPEC_MAXT=$budget SPEC_REPS=1
        if [[ $budget == 384 ]]; then export SPEC_REPS=3; fi
        out=perf/p8m/lookup/$budget
        bash perf/p8bc/probe.sh "$out" "$pass" "$width"
        for k in ${SPEC_KINDS//,/ }; do
            for ext in ids json; do cmp "$out/$pass.$k.$ext" "perf/p8m/matrix/main/$budget/plain.$k.$ext"; done
        done
    done
done
