#!/bin/bash
# Round 3: wall-time parts of a DSpark round (LLAMA_SPEC_PROF=1) on the spec_probe prompts.
# usage: perf/p6/r3_prof.sh <n_max> [kinds, default "edit"] [-- extra server args]; env IK_* / LLAMA_* pass through
cd "$(dirname "$0")/../.." && . perf/p5env.sh
N=$1; KINDS=${2:-edit}; shift; shift; [ "$1" = "--" ] && shift
L=${PROF_LOG:-/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/r3prof.log}
EXTRA="--backend-sampling"
[ "$N" != 0 ] && EXTRA="$EXTRA --spec-type draft-dspark --spec-draft-model ${DRAFT_FILE:-$DRAFT_FOLD} --spec-draft-n-max $N --spec-draft-p-min ${PMIN:-0}"
IK_MMA=${IK_MMA-3} IK_MR=${IK_MR-1} LLAMA_SPEC_PROF=${PROF-1} build/bin/llama-server -m $Q2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -np 1 --port 8043 --host 127.0.0.1 $EXTRA "$@" >$L 2>&1 &
for i in $(seq 1 240); do curl -sf -m1 http://127.0.0.1:8043/health >/dev/null && break; sleep 0.5; done
python3 - $KINDS <<'PY'
import json, sys, urllib.request, os
R = os.path.expanduser("~/llama.cpp-fast/perf/")
P = {"code": "Write a Python module that implements an LRU cache with per-entry time-to-live, thread safety and hit/miss statistics. Include unit tests.",
     "prose": "Write a 300-word story about a lighthouse keeper who finds a message in a bottle.",
     "edit": "Rewrite the following Python code so that every function and variable has type hints and the HTTP calls go through one helper function. Output the complete file, changing nothing else.\n\n```python\n" + open(R + "bench.py").read() + "\n" + open(R + "kernsum.py").read() + "\n```"}
for k in sys.argv[1:]:
    b = {"model": "x", "messages": [{"role": "user", "content": P[k]}], "max_tokens": 384, "temperature": 0, "seed": 1, "cache_prompt": False}
    t = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8043/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))["timings"]
    print(k, {x: t.get(x) for x in ("prompt_n", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")})
PY
kill -INT %1; wait
grep spec_prof $L
