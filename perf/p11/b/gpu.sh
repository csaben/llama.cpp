#!/usr/bin/env bash
# Run one GPU0 measurement from the repository root.
set -eu
cd "$(dirname "$0")/../../.."
lock=/home/arelius/llama.cpp-fast/perf/p11/gpu0.lock
mkdir -p build/p11
if [[ -z ${B_MIN_FREE_MIB+x} ]]; then
    case ${1##*/}:${2:-} in
        run.sh:mr|run.sh:ppl) B_MIN_FREE_MIB=11500 ;;
        run.sh:bench|run.sh:logits) B_MIN_FREE_MIB=14000 ;;
        run.sh:probe) B_MIN_FREE_MIB=18500 ;;
        long.sh:*) B_MIN_FREE_MIB=19000 ;;
        *) B_MIN_FREE_MIB=0 ;;
    esac
    if [[ ${B_MODEL:-} == *IQ4_NL* ]]; then B_MIN_FREE_MIB=$((B_MIN_FREE_MIB+6300)); fi
    B_MIN_FREE_MIB=$((B_MIN_FREE_MIB > 22500 ? 22500 : B_MIN_FREE_MIB))
fi
export B_MIN_FREE_MIB
# Queue one Stream B request at the shared lock. Release both locks before waiting for an unrelated GPU user.
while true; do
    if [[ -r build/p11/gpu-free.cache ]] && read -r observed available <build/p11/gpu-free.cache; then
        age=$(( $(date +%s) - observed ))
        if ((available < B_MIN_FREE_MIB && age >= 0 && age < 45)); then
            if [[ -e perf/STOP || -e /home/arelius/llama.cpp-fast/perf/STOP ]]; then exit 75; fi
            sleep "$((45 - age))"
        fi
    fi
    status=0
    flock build/p11/stream-b.lock flock "$lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 P8_GPU=0 bash -c '
        set -eu
        if [[ -e perf/STOP || -e /home/arelius/llama.cpp-fast/perf/STOP ]]; then exit 75; fi
        if [[ $(TZ=America/New_York date +%F) == 2026-11-27 && $(TZ=America/New_York date +%H%M) > 0729 ]]; then exit 75; fi
        free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
        echo "$(date +%s) $free" >build/p11/gpu-free.cache.tmp
        mv build/p11/gpu-free.cache.tmp build/p11/gpu-free.cache
        if ((free < B_MIN_FREE_MIB)); then exit 78; fi
        exec "$@"
    ' p11b "$@" || status=$?
    if ((status != 78)); then exit "$status"; fi
    sleep 45
done
