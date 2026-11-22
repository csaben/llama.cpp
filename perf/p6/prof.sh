#!/bin/bash
# usage: perf/p6/prof.sh <name> <rows> <llama-bench args...>   -> kernel totals of the run (nsys 2024.6, GPU1)
# e.g. perf/p6/prof.sh pp4 30 -p 4 -n 0 -r 20      (. perf/p5env.sh first)
NSYS=$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys
R=$(cd "$(dirname "$0")/../.." && pwd); O=${PROF_DIR:-/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/p6prof}
mkdir -p "$O"; N=$1; ROWS=$2; shift 2
rm -f "$O/$N".*
$NSYS profile -t cuda --cuda-graph-trace=node -o "$O/$N" --force-overwrite true "$R/build/bin/llama-bench" -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 "$@" >/dev/null 2>&1
$NSYS stats --report cuda_gpu_kern_sum --format csv -o "$O/$N" "$O/$N.nsys-rep" >/dev/null 2>&1
python3 "$R/perf/kernsum_total.py" "$O/${N}_cuda_gpu_kern_sum.csv" "$ROWS"
