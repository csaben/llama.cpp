#!/usr/bin/env bash
# Check the larger target microbatch against the same-flags plain server.
set -euo pipefail
export P8_GPU=1
source perf/p8bc/env.sh
export SPEC_KINDS=agent,edit@32K LLAMA_DSPARK_ADAPTIVE=1
export LLAMA_ARG_UBATCH=2048 LLAMA_ARG_SPEC_DRAFT_N_UBATCH=512
out=perf/p8m/split
mkdir -p "$out"
build/bin/llama-perplexity -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 2048 -b 2048 -ub 2048 --chunks 8 -f perf/ppl.txt >build/p8m/split-ppl.log 2>&1
rg 'Final estimate' build/p8m/split-ppl.log >"$out/ppl.txt"
for budget in 384 3000; do
    export SPEC_MAXT=$budget SPEC_REPS=1
    if [[ $budget == 384 ]]; then export SPEC_REPS=3; fi
    bash perf/p8bc/probe.sh "$out/$budget" plain 0
    bash perf/p8bc/probe.sh "$out/$budget" n6-adapt 6
    python3 perf/p6/r4_cmp.py "$out/$budget" plain n6-adapt >"$out/$budget/compare.txt"
done
