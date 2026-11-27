#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
shift
mkdir -p "$out"
for op in "$@"; do
    log="build/p11/ops-$(basename "$out")-$op.log"
    perf/p11/b/gpu.sh perf/p11/b/run.sh ops -o "$op" >"$log" 2>&1
    sed -E 's/\x1B\[[0-9;]*[mK]//g' "$log" | rg 'tests passed|Backend CUDA0:|FAIL' >"$out/$op.txt"
    rg -q '[1-9][0-9]*/[1-9][0-9]* tests passed' "$out/$op.txt"
    if rg -q 'FAIL' "$out/$op.txt"; then exit 1; fi
done
