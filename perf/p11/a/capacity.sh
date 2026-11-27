#!/usr/bin/env bash
# Probe one slot, one process, with the live vision projector attached.
set -euo pipefail
cd "$(dirname "$0")/../../.."
placement=$1 mode=$2 label=$3 ctx=$4 depth=$5 budget=$6
if [[ ${P11_CAP_LOCKED:-0} != 1 ]]; then
    exec perf/p11/a/with-gpus.sh "$placement" env P11_CAP_LOCKED=1 bash "$0" "$@"
fi
export P8_GPU=1
source perf/p8bc/env.sh
export LLAMA_DSPARK_ADAPTIVE=1
snap=${P11_BIN:-$HOME/inkling-live-bin-p9-72fd097}
export LD_LIBRARY_PATH="$snap:$PWD/build/bin"
model=${P11_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf}
args=() draft_args=() cards=1
if [[ $placement != one ]]; then
    export CUDA_VISIBLE_DEVICES=1,0 GGML_CUDA_P2P=1
    cards=0,1
    if [[ $placement == separate-both ]]; then
        args=(--device CUDA0,CUDA1 -ts 1/0)
        draft_args=(--spec-draft-device CUDA1 --spec-draft-override-tensor mask_embd.weight=CUDA1)
    elif [[ $placement == head || $placement == layer || $placement == table || $placement == balanced ]]; then
        args=(-ts 1/0)
        if [[ $placement == layer ]]; then args=(--device CUDA1,CUDA0 -ts 1/48); fi
        if [[ $placement == balanced ]]; then args=(--device CUDA1,CUDA0 -ts "${P11_LAYER_SPLIT:-1/3}"); fi
        if [[ $placement == table ]]; then export LLAMA_ARG_OVERRIDE_TENSOR='token_embd.weight=CUDA1'; fi
        draft_args=(--spec-draft-device CUDA0)
        export IK_PEER_HEAD=CUDA1
    else
        args=(--device CUDA0)
        draft_args=(--spec-draft-device CUDA0)
    fi
fi
if [[ $mode == draft ]]; then args+=(--spec-type draft-dspark --spec-draft-model "$P8_DRAFT" --spec-draft-n-max 6 --spec-draft-n-ubatch "${P11_DRAFT_UB:-512}" --spec-draft-p-min 0 "${draft_args[@]}"); fi
out=perf/p11/a/capacity/$label
mkdir -p perf/p11/a/capacity build/p11a/capacity
log=build/p11a/capacity/$label.server
ulimit -c 0
"$snap/llama-server" -m "$model" --mmproj "$HOME/models/inkling-2-mini/mmproj-Inkling-2-Mini-Q8_0.gguf" -c "$ctx" -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub "${P11_UB:-512}" --backend-sampling --host 127.0.0.1 --port 18502 --alias inkling-mini -lv 4 "${args[@]}" >"$log" 2>&1 &
pid=$!
nvidia-smi -i "$cards" --query-gpu=timestamp,index,memory.used,memory.free --format=csv,noheader,nounits -lms 250 >"build/p11a/capacity/$label.vram" &
monitor=$!
nvidia-smi -i "$cards" --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits -lms 250 >"build/p11a/capacity/$label.process-vram" &
process_monitor=$!
trap 'kill "$pid" "$monitor" "$process_monitor" 2>/dev/null || :; wait "$pid" || :; wait "$monitor" || :; wait "$process_monitor" || :' EXIT
for i in {1..240}; do
    kill -0 "$pid" || exit 1
    if curl -fsS --max-time 1 http://127.0.0.1:18502/health >/dev/null 2>&1; then break; fi
    sleep 0.5
done
curl -fsS --max-time 2 http://127.0.0.1:18502/health >/dev/null
if [[ -n ${P11_IMAGE:-} ]]; then
    python3 perf/p11/a/image-check.py "$P11_IMAGE" "$budget" "$out"
elif (( depth > 0 )); then
    python3 perf/p11/a/deep.py "$depth" "$budget" "$out"
fi
rg 'n_ctx_slot =|model buffer size =|KV buffer size =|compute buffer size =' "$log" >"$out.memory"
nvidia-smi -i "$cards" --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader | awk -F', ' -v p="$pid" '$2 == p' >>"$out.memory"
awk -F', ' '{d=$2; if ($3>peak[d]) peak[d]=$3; if (!(d in free) || $4<free[d]) free[d]=$4} END {for (d in peak) print "physical", d, "peak used MiB", peak[d], "min free MiB", free[d]}' "build/p11a/capacity/$label.vram" >>"$out.memory"
awk -F', ' -v pid="$pid" '$2 == pid && $3+0 > peak[$1]+0 {peak[$1]=$3+0} END {for (uuid in peak) print "process peak MiB", uuid, peak[uuid]}' "build/p11a/capacity/$label.process-vram" >>"$out.memory"
