#!/usr/bin/env bash
# Measure whole-device free memory alongside the uncached 64K-slot tool replay.
set -euo pipefail
export P8_GPU=1
label=${1:?run label}
out=perf/p8m/ttft/$label
mkdir -p "$out"
nvidia-smi -i 1 --query-gpu=timestamp,pci.bus_id,memory.used,memory.free --format=csv,nounits -lms 250 >"build/p8m/$label-memory.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || :; wait "$monitor" 2>/dev/null || :' EXIT
bash perf/p8bc/ttft.sh "p8m-$label"
cp "build/p8-prefill/p8m-$label/ttft.txt" "$out/samples.txt"
awk -F', ' 'NR == 2 {min = $4} NR > 1 {if ($4 < min) min = $4; if ($3 > peak) peak = $3} END {if (NR < 2) exit 1; print "peak used MiB", peak; print "minimum free MiB", min}' "build/p8m/$label-memory.csv" >"$out/memory.txt"
