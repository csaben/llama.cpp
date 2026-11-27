#!/usr/bin/env bash
set -euo pipefail
repo=/home/arelius/llama.cpp-fast
placement=$1
shift
mkdir -p "$repo/build/p11a"
local_lock=$repo/build/p11a/gpu1.lock
active_lock=$repo/build/p11a/gpu1-use.lock
turn_lock=$repo/build/p11a/gpu1-turn.lock
shared_lock=$repo/perf/p11/gpu0.lock
guard='
    [[ ! -e "$1/perf/STOP" ]] || exit 75
    shift
    remaining=$(( $(date -d "${P11_UNTIL:-2026-11-27 07:30 EST}" +%s) - $(date +%s) ))
    ((remaining > 4)) || exit 75
    if ((remaining < 900)); then exec timeout -k 2 "$((remaining-3))" "$@"; fi
    exec "$@"
'
if [[ $placement == one ]]; then
    exec flock "$local_lock" flock -s "$turn_lock" flock "$active_lock" env P11_PRIMARY_READY=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash -c "$guard" guard "$repo" "$@"
fi
# Leave GPU1 available while Stream B works. Release GPU0 at once if a one-card check is running.
if flock "$shared_lock" flock -n -E 73 "$active_lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 bash -c "$guard" guard "$repo" "$@"; then
    exit 0
else
    rc=$?
    [[ $rc == 73 ]] || exit "$rc"
fi
# After a collision, reserve the next GPU1 turn before rejoining the shared queue.
exec flock -x "$turn_lock" flock "$shared_lock" flock -n -E 73 "$active_lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 bash -c "$guard" guard "$repo" "$@"
