#!/usr/bin/env bash
# Switch the Inkling service to the Phase 9 build after its GPU1 checks have passed.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
source perf/p5env.sh
unit=inkling-3090-experiment
draft="$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf"
test -r "$Q2"
test -r "$draft"
snapshot="$HOME/inkling-live-bin-p9-72fd097"
test -x "$snapshot/llama-server"
(cd "$snapshot" && sha256sum -c "$repo/perf/p9/snapshot.sha256")
if systemctl --user is-active --quiet "$unit"; then
    systemctl --user stop "$unit"
fi
if systemd-run --user --unit="$unit" --working-directory="$repo" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES=1 --setenv=LD_LIBRARY_PATH="$snapshot" \
    --setenv=LLAMA_INKLING_CTRL_F16=1 --setenv=IK_BATCH_MESSAGES=1 \
    --setenv=IK_SEGMENTED_Q6=1 --setenv=IK_SEGMENTED_IQ4=1 \
    --setenv=IK_CTRL_APPLY8=1 --setenv=IK_MIX_SPECIALIZE=1 \
    --setenv=IK_MR=1 --setenv=IK_MMA=3 --setenv=IK_WINDOW_COLS=1 --setenv=IK_QUEUE_INPUTS=1 \
    --setenv=LLAMA_DSPARK_DEV=1 --setenv=LLAMA_DSPARK_ROUND=1 --setenv=LLAMA_DSPARK_GRAPHS=1 \
    --setenv=LLAMA_DSPARK_ADAPTIVE=1 --setenv=LLAMA_DSPARK_POLICY=1 --setenv=GGML_CUDA_GRAPH_CHUNK_SIZE=0 \
    "$snapshot/llama-server" -m "$Q2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -c 65536 -np 1 -b 2048 -ub 512 --override-tensor token_embd.weight=CUDA0 --backend-sampling --host 127.0.0.1 --port 8031 --alias inkling-mini \
    --spec-type draft-dspark --spec-draft-model "$draft" --spec-draft-override-tensor mask_embd.weight=CUDA0 --spec-draft-n-ubatch 512 --spec-draft-n-max 6 --spec-draft-p-min 0; then
    for ((i=0; i<120; i++)); do
        if curl -fsS -m 2 http://127.0.0.1:8031/health >/dev/null 2>&1; then
            echo "Phase 9 ready on :8031 from $snapshot"
            exit 0
        fi
        sleep 2
    done
fi
echo "Phase 9 startup failed. Restoring phase 5." >&2
bash perf/live/restore-p5.sh
exit 1
