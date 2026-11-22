#!/usr/bin/env bash
# Compare one environment option in the same binary, in off/on/on/off order.
set -euo pipefail
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0
export LD_LIBRARY_PATH=${R4_BIN:-"$PWD/build/bin"}
bench_exe=build/bin/llama-bench
if [[ -n ${R4_BIN:-} && -x $R4_BIN/llama-bench ]]; then bench_exe=$R4_BIN/llama-bench; fi
label=$1 depth=$2 flag=$3 on=$4 off=${5:-0} repeats=${6:-3}
for value in "$off" "$on" "$on" "$off"; do
    if [[ -n ${R4_MAX_TEMP:-} ]]; then
        while (( $(nvidia-smi -i 00000000:01:00.0 --query-gpu=temperature.gpu --format=csv,noheader,nounits) > R4_MAX_TEMP )); do sleep 2; done
    fi
    echo "$flag=$value start_C=$(nvidia-smi -i 00000000:01:00.0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)"
    env "$flag=$value" "$bench_exe" -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d "$depth" -r "$repeats" 2>>"build/p7l3r4/$label.log" | rg 'tg128|build:'
done >"perf/p7l3r4/$label.txt"
