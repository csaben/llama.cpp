#!/usr/bin/env bash
# Run one GPU0 measurement from the repository root.
set -eu
cd "$(dirname "$0")/../../.."
lock=/home/arelius/llama.cpp-fast/perf/p11/gpu0.lock
exec flock "$lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 P8_GPU=0 bash -c '
    set -eu
    if [[ -e perf/STOP || -e /home/arelius/llama.cpp-fast/perf/STOP ]]; then exit 75; fi
    if [[ $(TZ=America/New_York date +%F) == 2026-11-27 && $(TZ=America/New_York date +%H%M) > 0729 ]]; then exit 75; fi
    exec "$@"
' p11b "$@"
