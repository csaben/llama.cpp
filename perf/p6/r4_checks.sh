#!/usr/bin/env bash
# Replay the failing trace and exercise mixed-width reject/rollback steps at long prefixes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source perf/p5env.sh
export GPU=${CHECKS_GPU:-$GPU} CUDA_VISIBLE_DEVICES=${CHECKS_GPU:-$CUDA_VISIBLE_DEVICES}
export LD_LIBRARY_PATH=$PWD/build/bin
out=${CHECKS_OUT:-perf/p6/r4_checks}
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
gzip -dc perf/p6/r4_repro/n3.trace.gz > "$tmp/n3.trace"
g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p6/mr_check.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o build/bin/mr_check
(
    export IK_MR=1 IK_MMA=3 MR_RS_A=0 MR_RS=4
    MR_RS=3 MR_CTX=16384 MR_TRACE="$tmp/n3.trace" build/bin/mr_check "$Q2" perf/ppl.txt
    for prefix in 1533 3069 4500; do
        MR_RAND=2 MR_SIZES=5 MR_PLAIN_PCT=0 build/bin/mr_check "$Q2" perf/ppl.txt "$prefix" 24
    done
    IK_ATTN_FIXED_MR=0 MR_RAND=2 MR_SIZES=5 build/bin/mr_check "$Q2" perf/ppl.txt 3069 24
    for prefix in 2000 2046 2558 3069 4094 4500; do
        MR_RAND=37 MR_SIZES=2,3,4,5 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 build/bin/mr_check "$Q2" perf/ppl.txt "$prefix" 64
    done
) > "$out/rows.txt" 2>&1
args=(-m "$Q2" -ngl 99 -ctk q8_0 -ctv q8_0)
for label in plain plain-mr; do
    flags=()
    if [[ $label == plain-mr ]]; then flags=(IK_MR=1 IK_MMA=3); fi
    env "${flags[@]}" build/bin/llama-bench "${args[@]}" -fa 1 -p 0 -n 128 -d 0,4096,16384,32768 -r 3 > "$out/$label.bench.txt" 2> "$tmp/bench.log"
    for mode in prefill decode; do
        params=(-c 2048 -b 2048 --chunks 8)
        if [[ $mode == decode ]]; then params=(-c 1024 -b 1024 -ub 1 --chunks 2); fi
        env "${flags[@]}" build/bin/llama-perplexity "${args[@]}" -fa on -f perf/ppl.txt "${params[@]}" > "$tmp/ppl.log" 2>&1
        grep 'Final estimate' "$tmp/ppl.log" > "$out/$label.$mode.txt"
    done
    env "${flags[@]}" MODEL="$Q2" bash perf/greedy.sh "$out/$label" -ctk q8_0 -ctv q8_0
done
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_ID,INKLING_ROUTER,INKLING_ATTN > "$tmp/ops.log" 2>&1
grep -E 'tests passed|Backend CUDA0: |FAIL' "$tmp/ops.log" > "$out/ops.txt"
