#!/usr/bin/env bash
# Invoke through gpu.sh. Set B_LIBS to the runtime under test.
set -euo pipefail
[[ ${CUDA_DEVICE_ORDER:?} == PCI_BUS_ID && ${CUDA_VISIBLE_DEVICES:?} == 0 && ${P8_GPU:?} == 0 ]]
window_cols=${B_WINDOW_COLS:-${IK_WINDOW_COLS:-1}}
source perf/p8bc/env.sh
export GPU=0 CUDA_VISIBLE_DEVICES=0 SPEC_PORT=18550 SPEC_CTX=${B_CTX:-131072}
export MODEL=${B_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf}
export LD_LIBRARY_PATH="${B_LIBS:-$HOME/inkling-live-bin-p9-72fd097}:$PWD/build/p11/p9-tools"
export SPEC_SERVER_BIN="${LD_LIBRARY_PATH%%:*}/llama-server"
export LLAMA_DSPARK_ADAPTIVE=${B_ADAPT:-1} LLAMA_DSPARK_POLICY=${B_POLICY:-1}
export IK_WINDOW_COLS=$window_cols IK_MMA=${B_MMA:-3} IK_MR=${B_MR:-1}
if [[ ${B_PLAIN:-0} == 1 ]]; then
    unset IK_MR IK_MMA IK_WINDOW_COLS IK_QUEUE_INPUTS GGML_CUDA_GRAPH_CHUNK_SIZE
fi
ubatch=${B_UBATCH:-512}
args=(-m "$MODEL" -ngl 99 -ctk q8_0 -ctv q8_0)
bench=(build/p11/p9-tools/llama-bench "${args[@]}" -fa 1 -ot token_embd.weight=CUDA0 -b 2048 -ub "$ubatch" -o json)
mode=$1
shift
case $mode in
    bench) exec "${bench[@]}" "$@" ;;
    probe)
        out=$1 label=$2 width=$3
        export SPEC_KINDS=$4 SPEC_REPS=$5 SPEC_MAXT=$6 SPEC_TOKEN_IDS=1
        mkdir -p "$out"
        export SPEC_SERVER_LOG="$out/$label.server.log"
        extra=()
        if ((width)); then extra+=(--spec-draft-n-ubatch 512); fi
        exec bash perf/p8bc/probe.sh "$out" "$label" "$width" -b 2048 -ub "$ubatch" --alias inkling-mini "${extra[@]}" "${@:7}" ;;
    logits) exec build/p11/p9-tools/logits "$MODEL" perf/ppl.txt "$@" ;;
    ops) exec build/bin/test-backend-ops test -b CUDA0 "$@" ;;
    ppl) exec "${B_PPL_TOOL:-build/p11/p9-tools/llama-perplexity}" "${args[@]}" -fa on -f perf/ppl.txt "$@" ;;
    mr) exec "${B_MR_TOOL:-build/p11/p9-tools/mr_check}" "$MODEL" "$@" ;;
    *) exit 2 ;;
esac
