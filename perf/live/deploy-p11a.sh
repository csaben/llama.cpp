#!/usr/bin/env bash
# Manual deployment after the shared GPU benchmark window.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
source perf/p5env.sh
unit=inkling-3090-experiment
draft="$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf"
mmproj="$HOME/models/inkling-2-mini/mmproj-Inkling-2-Mini-Q8_0.gguf"
Q2="$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf"
fallback() {
    exec bash perf/live/deploy-p9v-uncensored-128k.sh
}
read -r short mode < perf/p11/a/release.txt || fallback
[[ $short =~ ^[a-f0-9]{8}$ ]] || fallback
snapshot="$HOME/inkling-live-bin-p11a-$short"
if [[ ! -r $mmproj || ! -r $draft || ! -r $Q2 || ! -x $snapshot/llama-server ]] ||
        ! (cd "$snapshot" && sha256sum --strict -c "$repo/perf/p11/a/snapshot.sha256"); then
    fallback
fi
devices=1
peer_env=() placement=() draft_placement=()
case $mode in
    head|direct)
        devices=1,0
        peer_env=(--setenv=GGML_CUDA_P2P=1 --setenv=IK_PEER_HEAD=CUDA1 --setenv=IK_PEER_NATIVE=1)
        if [[ $mode == direct ]]; then peer_env+=(--setenv=IK_PEER_DIRECT=1); fi
        placement=(-ts 1/0)
        draft_placement=(--spec-draft-device CUDA0);;
    one) ;;
    *) fallback;;
esac
if systemctl --user is-active --quiet "$unit"; then
    systemctl --user stop "$unit"
fi
if systemd-run --user --collect --unit="$unit" --working-directory="$repo" \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES="$devices" --setenv=LD_LIBRARY_PATH="$snapshot" "${peer_env[@]}" \
    --setenv=LLAMA_INKLING_CTRL_F16=1 --setenv=IK_BATCH_MESSAGES=1 \
    --setenv=IK_SEGMENTED_Q6=1 --setenv=IK_SEGMENTED_IQ4=1 \
    --setenv=IK_CTRL_APPLY8=1 --setenv=IK_MIX_SPECIALIZE=1 \
    --setenv=IK_MR=1 --setenv=IK_MMA=3 --setenv=IK_WINDOW_COLS=1 --setenv=IK_QUEUE_INPUTS=1 \
    --setenv=LLAMA_DSPARK_DEV=1 --setenv=LLAMA_DSPARK_ROUND=1 --setenv=LLAMA_DSPARK_GRAPHS=1 \
    --setenv=LLAMA_DSPARK_ADAPTIVE=1 --setenv=LLAMA_DSPARK_POLICY=1 --setenv=GGML_CUDA_GRAPH_CHUNK_SIZE=0 \
    "$snapshot/llama-server" -m "$Q2" --mmproj "$mmproj" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -c 131072 -np 1 -b 2048 -ub 512 "${placement[@]}" --override-tensor token_embd.weight=CUDA0 --backend-sampling --host 127.0.0.1 --port 8031 --alias inkling-mini \
    --spec-type draft-dspark --spec-draft-model "$draft" "${draft_placement[@]}" --spec-draft-override-tensor mask_embd.weight=CUDA0 --spec-draft-n-ubatch 512 --spec-draft-n-max 6 --spec-draft-p-min 0; then
    for ((i=0; i<120; i++)); do
        if curl -fsS -m 2 http://127.0.0.1:8031/health >/dev/null 2>&1; then
            echo "Phase 11A vision (uncensored 2-bit) ready on :8031 from $snapshot"
            exit 0
        fi
        sleep 2
    done
fi
echo "Phase 11A startup failed. Restoring the p9 uncensored 128K server." >&2
systemctl --user stop "$unit" || :
systemctl --user reset-failed "$unit" 2>/dev/null || :
fallback
