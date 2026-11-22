#!/usr/bin/env bash
set -eu -o pipefail
. perf/p8bc/env.sh
if [[ -n ${P8_LIBS:-} ]]; then
    export LD_LIBRARY_PATH="$P8_LIBS:$LD_LIBRARY_PATH"
elif [[ $1 == base || $1 == control* ]]; then
    for lib in libggml-cuda.so.1 libllama.so.1 libllama-common.so.1; do
        test -s "build/p8-b-final/$lib"
    done
    export LD_LIBRARY_PATH="$PWD/build/p8-b-final:$LD_LIBRARY_PATH"
fi
label=${1:?}
out=build/p8-prefill/$label
mkdir -p "$out"
args=(-m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0)
build/bin/llama-bench "${args[@]}" -p 4096,16384,32768 -n 0 -b 2048 -ub "${UB:-512}" -r "${REPS:-3}" -o json >"$out/bench.json" 2>"$out/bench.log"
if [[ ${PROFILE:-0} == 1 ]]; then
    for p in 4096 16384 32768; do
        PROF_DIR="$PWD/$out" bash perf/p6/prof.sh "pp$p" 24 --no-warmup -p "$p" -n 0 -b 2048 -ub "${UB:-512}" -r 1 >"$out/kernels-$p.txt"
    done
fi
if [[ ${PPL:-0} == 1 ]]; then
    build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 2048 -b 2048 -ub "${UB:-512}" --chunks 8 -f perf/ppl.txt >"$out/ppl.log" 2>&1
    rg 'Final estimate' "$out/ppl.log"
fi
