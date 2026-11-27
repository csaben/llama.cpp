#!/usr/bin/env bash
# Run one probe per process. CUDA0 is always physical GPU1 for the target.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?bench, draft, or plain}
placement=${2:?one, row, tensor, separate, or peer}
label=${3:?output label}
shift 3
export P8_GPU=1
if [[ ${P11_LOCKED:-} != 1 ]]; then
    exec perf/p11/a/with-gpus.sh "$placement" env P11_LOCKED=1 bash "$0" "$mode" "$placement" "$label" "$@"
fi
[[ ! -e perf/STOP ]] || exit 75
source perf/p8bc/env.sh
export MODEL="${P11_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf}"
export SPEC_PORT=18501 SPEC_CTX=${P11_CTX:-131072}
export SPEC_KINDS="${P11_KINDS:-edit}"
export SPEC_SERVER_BIN="${P11_BIN:-$HOME/inkling-live-bin-p9-72fd097}/llama-server"
export LD_LIBRARY_PATH="$(dirname "$SPEC_SERVER_BIN"):$PWD/build/bin"
export SPEC_SERVER_LOG="${P11_LOG:-$PWD/build/p11a/$label.server}"
export SPEC_OUT="${P11_OUT:-$PWD/perf/p11/a/screen}"
mkdir -p "$SPEC_OUT" build/p11a
ulimit -c 0
args=() draft_args=()
if [[ $placement != one ]]; then
    export CUDA_VISIBLE_DEVICES=1,0 GPU=1,0 GGML_CUDA_P2P=1
    case $placement in
        row|tensor) args=(-sm "$placement" -ts 1/1);;
        separate) args=(--device CUDA0); draft_args=(--spec-draft-device CUDA1 --spec-draft-override-tensor mask_embd.weight=CUDA1);;
        separate-both) args=(--device CUDA0,CUDA1 -ts 1/0); draft_args=(--spec-draft-device CUDA1 --spec-draft-override-tensor mask_embd.weight=CUDA1);;
        head) args=(-ts 1/0); draft_args=(--spec-draft-device CUDA0); export IK_PEER_HEAD=CUDA1;;
        peer) args=(--device CUDA0); draft_args=(--spec-draft-device CUDA0);;
        *) exit 2;;
    esac
fi
if [[ $mode == bench ]]; then
    unset IK_MR IK_MMA
    build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ot token_embd.weight=CUDA0 -b 2048 -ub 512 -o json "${args[@]}" "$@" >"$SPEC_OUT/$label.json" 2>"$SPEC_SERVER_LOG"
else
    export LLAMA_DSPARK_POLICY=${P11_POLICY:-1}
    if [[ ${P11_ADAPT:-1} == 1 ]]; then export LLAMA_DSPARK_ADAPTIVE=1; else unset LLAMA_DSPARK_ADAPTIVE; fi
    args+=(--backend-sampling -b 2048 -ub 512 --alias inkling-mini)
    if [[ $mode == draft ]]; then
        args+=(--spec-type draft-dspark --spec-draft-model "$P8_DRAFT" --spec-draft-n-max "${P11_WIDTH:-6}" --spec-draft-n-ubatch 512 --spec-draft-p-min 0 "${draft_args[@]}")
    else
        [[ $mode == plain ]]
    fi
    perf/spec_probe.py "$label" -- "${args[@]}" "$@" | tee "$SPEC_OUT/$label.txt"
fi
