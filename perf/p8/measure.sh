#!/usr/bin/env bash
# Run from the repository root with two runtime library directories.
set -euo pipefail
source perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
label=${2:?label} depths=${3:?depths} control=${4:?control library directory}
out=perf/p8/$label
work=build/p8
mkdir -p "$out" "$work"
bench=(build/bin/llama-bench -m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128)
case "$1" in
    pair)
        candidate=${5:?candidate library directory}
        for pass in A1 B1 B2 A2; do
            export LD_LIBRARY_PATH="$control"
            if [[ $pass == B* ]]; then export LD_LIBRARY_PATH="$candidate"; fi
            "${bench[@]}" -d "$depths" -r 3 >"$out/$pass.txt" 2>"$work/$label-$pass.log"
        done ;;
    trace)
        export LD_LIBRARY_PATH="$control"
        for depth in ${depths//,/ }; do
            "$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys" profile --force-overwrite=true -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --export=sqlite -o "$work/$label-$depth" "${bench[@]}" -d "$depth" -r 1 >"$work/$label-$depth.log" 2>&1
        done
        for mode in first single mean; do
            args=()
            case $mode in first) args=(--step 0);; mean) args=(--mean);; esac
            for depth in ${depths//,/ }; do
                python3 perf/p7l3r4/latency.py "$work/$label-$depth.sqlite" "$depth" "${args[@]}"
                echo
            done >"$out/latency-$mode.md"
        done ;;
    *) exit 2 ;;
esac
