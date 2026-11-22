#!/usr/bin/env bash
set -euo pipefail
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 LD_LIBRARY_PATH="$PWD/build/bin"
work=build/p7l3r4
out=perf/p7l3r4
mkdir -p "$work" "$out"
bench=(-m "$Q2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128)
case $1 in
trace)
    "$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys" profile -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --export=sqlite -o "$work/$2" build/bin/llama-bench "${bench[@]}" -d "$3" -r 1 >"$work/$2.log" 2>&1 ;;
bench)
    bin=${4:-"$PWD/build/bin"}
    while (( $(nvidia-smi -i 00000000:01:00.0 --query-gpu=temperature.gpu --format=csv,noheader,nounits) > 63 )); do sleep 2; done
    LD_LIBRARY_PATH="$bin" "$bin/llama-bench" "${bench[@]}" -d "$3" -r "${5:-3}" >"$out/$2.txt" 2>"$work/$2.log" ;;
*) exit 2 ;;
esac
