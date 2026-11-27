#!/usr/bin/env bash
set -euo pipefail
repo=/home/arelius/llama.cpp-fast
placement=$1
shift
mkdir -p "$repo/build/p11a"
local_lock=$repo/build/p11a/gpu1.lock
shared_lock=$repo/perf/p11/gpu0.lock
guard='[[ ! -e "$1/perf/STOP" && $(date +%s) -lt $(date -d "${P11_UNTIL:-2026-11-27 07:30 EST}" +%s) ]] || exit 75; shift; exec "$@"'
if [[ $placement == one ]]; then
    exec flock "$local_lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash -c "$guard" guard "$repo" "$@"
fi
# Queue one A check at a time. Only request GPU0 once GPU1 belongs to this check.
exec flock "$local_lock" flock "$shared_lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 bash -c "$guard" guard "$repo" "$@"
