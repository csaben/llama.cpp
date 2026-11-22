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
out=build/p8-prefill/${1:?}
mkdir -p "$out"
args=()
if (( ${DRAFT_WIDTH:-0} )); then
    export LLAMA_DSPARK_ADAPTIVE=1
    args+=(--spec-type draft-dspark --spec-draft-model "$P8_DRAFT" --spec-draft-n-max "$DRAFT_WIDTH" --spec-draft-p-min 0)
fi
# Warm the same server before timing three uncached requests at each context.
build/bin/llama-server -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 1 --backend-sampling "${args[@]}" -b 2048 -ub "${UB:-512}" --port 18432 --host 127.0.0.1 >"$out/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || :; wait "$server" || :' EXIT
for ((i=0; i<120; i++)); do
    if curl -fsS --max-time 1 http://127.0.0.1:18432/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server" 2>/dev/null; then tail -10 "$out/server.log" >&2; exit 1; fi
    sleep 1
done
python3 perf/p8bc/ttft.py | tee "$out/ttft.txt"
