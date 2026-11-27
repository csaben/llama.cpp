#!/usr/bin/env bash
set -euo pipefail
repo=/home/arelius/llama.cpp-fast
placement=$1
shift
mkdir -p "$repo/build/p11a"
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
    exec flock "$turn_lock" flock "$active_lock" env P11_PRIMARY_READY=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash -c "$guard" guard "$repo" "$@"
fi
rank=(env P11_PRIMARY_READY=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 bash -c "$guard" guard "$repo")
reserve=12
while [[ ! -e $repo/perf/STOP ]]; do
    if flock "$shared_lock" flock -n -E 73 "$active_lock" "${rank[@]}" "$@"; then exit 0; else rc=$?; fi
    [[ $rc == 73 ]] || exit "$rc"

    # Give one-card work a turn if GPU0 stays busy. Increase the reservation after repeated collisions.
    if flock "$turn_lock" flock -E 74 -w "$reserve" "$shared_lock" flock -n -E 73 "$active_lock" "${rank[@]}" "$@"; then exit 0; else rc=$?; fi
    [[ $rc == 74 || $rc == 73 ]] || exit "$rc"
    reserve=$((2*reserve))
    [[ $(date +%s) -lt $(date -d "${P11_UNTIL:-2026-11-27 07:30 EST}" +%s) ]] || exit 75
done
exit 75
