#!/usr/bin/env bash
set -eu
d=perf/p11/b/baseline
run() { perf/p11/b/gpu.sh perf/p11/b/run.sh "$@"; }
for depth in 0 4096 16384 32768 65536; do
    run bench -p 0 -n 128 -d "$depth" -r 3 >"$d/tg-$depth.json" 2>"$d/tg-$depth.log"
done
for depth in 4096 16384 65536; do
    run bench -p "$depth" -n 0 -r 3 >"$d/pp-$depth.json" 2>"$d/pp-$depth.log"
done
for width in 6 0; do
    run probe "$d/short" "n$width" "$width" code,edit,prose,agent 1 3000
done
SPEC_KINDS=code,edit,prose,agent SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$d/short" n0 n6 >"$d/short/equality.txt"
