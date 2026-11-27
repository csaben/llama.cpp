#!/usr/bin/env bash
# Run one rollback, perplexity, logit dump, or backend check under its GPU lease.
set -euo pipefail
cd "$(dirname "$0")/../../.."
placement=$1 kind=$2 label=$3
shift 3
if [[ ${P11_CHECK_LOCKED:-0} != 1 ]]; then
    exec perf/p11/a/with-gpus.sh "$placement" env P11_CHECK_LOCKED=1 bash "$0" "$placement" "$kind" "$label" "$@"
fi
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MR_PRIMARY_DEVICE=CUDA0
export LD_LIBRARY_PATH="${P11_BIN:-$HOME/inkling-live-bin-p9-72fd097}:$PWD/build/bin"
model="${P11_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf}"
export IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0
args=()
if [[ $placement != one ]]; then
    export CUDA_VISIBLE_DEVICES=1,0 GGML_CUDA_P2P=1
    args=(--device CUDA0)
fi
if [[ ${P11_MR:-0} == 1 ]]; then export IK_MR=1 IK_MMA=3; fi
out="${P11_CHECK_OUT:-perf/p11/a/checks}"
mkdir -p "$out" build/p11a/checks
log="$PWD/build/p11a/checks/$label.log"
case $kind in
    prefill|decode)
        opts=(-c 2048 -b 2048 --chunks 8)
        if [[ $kind == decode ]]; then opts=(-c 1024 -b 1024 -ub 1 --chunks 2); fi
        build/bin/llama-perplexity -m "$model" -ngl 99 -ctk q8_0 -ctv q8_0 -fa on -f perf/ppl.txt "${opts[@]}" "${args[@]}" >"$log" 2>&1
        rg 'Final estimate' "$log" >"$out/$label.txt";;
    mr)
        build/p11a/mr_check "$model" "${P11_CORPUS:-perf/ppl.txt}" "$@" >"$log" 2>&1
        rg '0 rows differ from single-token decode, max \|logit diff\| 0$' "$log" >"$out/$label.txt";;
    logits)
        "${P11_LOGITS_BIN:-build/p11a/logits}" "$model" perf/ppl.txt "$@" >"$log.bin" 2>"$log"
        test -s "$log.bin"
        sha256sum "$log.bin" >"$out/$label.txt";;
    ops)
        build/bin/test-backend-ops test -b CUDA0 "$@" >"$log" 2>&1
        rg 'tests passed|Backend CUDA0:.*OK' "$log" >"$out/$label.txt";;
    *) exit 2;;
esac
