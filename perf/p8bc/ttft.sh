#!/usr/bin/env bash
set -eu -o pipefail
. perf/p8bc/env.sh
if [[ $1 == base || $1 == control* ]]; then export LD_LIBRARY_PATH="$PWD/build/p8-b-final:$LD_LIBRARY_PATH"; fi
if [[ -n ${P8_LIBS:-} ]]; then export LD_LIBRARY_PATH="$P8_LIBS:$PWD/build/bin"; fi
out=build/p8-prefill/${1:?}
mkdir -p "$out"
# Warm the same server before timing three uncached requests at each context.
build/bin/llama-server -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 65536 -np 1 --backend-sampling -b 2048 -ub "${UB:-512}" --port 18432 --host 127.0.0.1 >"$out/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || :; wait "$server" || :' EXIT
for ((i=0; i<120; i++)); do
    if curl -fsS --max-time 1 http://127.0.0.1:18432/health >/dev/null 2>&1; then break; fi
    kill -0 "$server" && sleep 1
done
python3 perf/p8bc/ttft.py | tee "$out/ttft.txt"
