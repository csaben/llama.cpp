#!/bin/bash
# usage: perf/p6/prof_server.sh <name> <rows> <kind code|edit|prose> [extra llama-server args]  (. perf/p5env.sh first)
# nsys kernel totals of one 256-token chat request through a fresh server on :8042.
NSYS=$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys
R=$(cd "$(dirname "$0")/../.." && pwd); O=${PROF_DIR:-/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/p6prof}
mkdir -p "$O"; N=$1; ROWS=$2; KIND=$3; shift 3
rm -f "$O/$N".*
$NSYS profile -t cuda --cuda-graph-trace=node -o "$O/$N" --force-overwrite true "$R/build/bin/llama-server" -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -np 1 --port 8042 --host 127.0.0.1 "$@" >"$O/$N.log" 2>&1 &
for i in $(seq 1 240); do curl -sf -m1 http://127.0.0.1:8042/health >/dev/null && break; sleep 0.5; done
python3 - "$KIND" <<'PY'
import json, sys, urllib.request, os
P = {"code": "Write a Python module that implements an LRU cache with per-entry time-to-live, thread safety and hit/miss statistics. Include unit tests.",
     "prose": "Write a 300-word story about a lighthouse keeper who finds a message in a bottle."}
P["edit"] = "Rewrite the following Python code so that every function and variable has type hints. Output the complete file, changing nothing else.\n\n```python\n" + open(os.path.expanduser("~/llama.cpp-fast/perf/bench.py")).read() + "\n```"
b = {"model": "x", "messages": [{"role": "user", "content": P[sys.argv[1]]}], "max_tokens": 256, "temperature": 0, "seed": 1, "cache_prompt": False}
t = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8042/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))["timings"]
print({k: t.get(k) for k in ("prompt_n", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")})
PY
pkill -INT -x llama-server; wait
$NSYS stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv -o "$O/$N" "$O/$N.nsys-rep" >/dev/null 2>&1
python3 "$R/perf/kernsum_total.py" "$O/${N}_cuda_gpu_kern_sum.csv" "$ROWS"
