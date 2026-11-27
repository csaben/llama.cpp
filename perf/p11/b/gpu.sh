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
queue=(flock build/p11/stream-b.lock)
if ((B_MIN_FREE_MIB >= 11000)); then
    queue=(flock "build/p11/stream-b-resident-$((PPID % 6)).lock")
    if [[ ${2:-} == probe ]]; then
        case ${3:-} in
            build/p11/matrix-narrow/*)
                suite=${3%/*}
                queue=(flock "build/p11/stream-b-narrow-${suite##*/}.lock") ;;
            build/p11/matrix-p9/main/*) queue=(flock build/p11/stream-b-p9-main.lock) ;;
            build/p11/matrix-p9/*) queue=(flock build/p11/stream-b-p9-extra.lock) ;;
        esac
    fi
    if [[ ${2:-} == bench && ${B_BENCH_GRID_LANE:-} =~ ^(timed|output)$ ]]; then
        queue=(flock "build/p11/stream-b-bench-${B_BENCH_GRID_LANE}.lock")
    fi
elif [[ ${B_OPERATOR_SWEEP:-0} == 1 ]]; then
    queue=(flock "build/p11/stream-b-operator-$((PPID % 4)).lock")
fi
if [[ ${B_LOCAL_GROUP_FD:-} =~ ^[0-9]+$ && /proc/self/fd/$B_LOCAL_GROUP_FD -ef build/p11/stream-b.lock ]]; then
    flock -n "$B_LOCAL_GROUP_FD"
    queue=()
fi
# Recheck free memory under the shared lock. Release both locks before waiting for an unrelated GPU user.
while true; do
    if [[ -r build/p11/gpu-free.cache ]] && read -r observed available <build/p11/gpu-free.cache; then
        age=$(( $(date +%s) - observed ))
        if ((available < B_MIN_FREE_MIB && age >= 0 && age < 45)); then
            if [[ -e perf/STOP || -e /home/arelius/llama.cpp-fast/perf/STOP ]]; then exit 75; fi
            sleep "$((45 - age))"
        fi
    fi
    status=0
    "${queue[@]}" flock "$lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 P8_GPU=0 bash -c '
        set -eu
        if [[ -e perf/STOP || -e /home/arelius/llama.cpp-fast/perf/STOP ]]; then exit 75; fi
        if [[ $(TZ=America/New_York date +%F) == 2026-11-27 ]]; then
            remaining=$(( $(TZ=America/New_York date -d "2026-11-27 07:30" +%s) - $(date +%s) ))
            if ((remaining <= 4)); then exit 75; fi
        fi
        free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
        echo "$(date +%s) $free" >build/p11/gpu-free.cache.tmp
        mv build/p11/gpu-free.cache.tmp build/p11/gpu-free.cache
        if ((free < B_MIN_FREE_MIB)); then exit 78; fi
        if ((${remaining:-86400} < 900)); then exec timeout -k 2 "$((remaining-3))" "$@"; fi
        exec "$@"
    ' p11b "$@" || status=$?
    if ((status != 78)); then exit "$status"; fi
    sleep 45
done
