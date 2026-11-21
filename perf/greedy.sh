#!/bin/bash
# usage: perf/greedy.sh <outdir> [extra llama-completion args]
# greedy (temp 0) generation of 3 fixed prompts, 200 tokens each, on GPU1
set -e
OUT=$1; shift
mkdir -p "$OUT"
M=${MODEL:-~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf}
BIN=${BIN:-build/bin/llama-completion}
i=0
while IFS= read -r p; do
  i=$((i+1))
  CUDA_VISIBLE_DEVICES=${GPU:-0} $BIN -m $M -ngl 99 -fa on -c 4096 -n 200 --temp 0 --top-k 1 -s 1 -no-cnv --no-warmup -p "$p" "$@" 2>/dev/null > "$OUT/gen$i.txt"
  echo "gen$i: $(wc -w < "$OUT/gen$i.txt") words"
done <<'PROMPTS'
Write a Python function that parses an ISO-8601 duration like 'P3DT4H5M' into seconds, with tests.
The history of the bicycle begins in the early nineteenth century, when
Explain step by step how TCP slow start and congestion avoidance work.
PROMPTS
