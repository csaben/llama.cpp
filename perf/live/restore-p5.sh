#!/bin/bash
# Restart the live endpoint from the saved phase-5 binaries in ~/inkling-live-bin-p5 WITHOUT copying build/ (build/ may hold experimental code).
# Move the live Inkling endpoint (localhost:8031, 2-bit model, 64K ctx) to the phase-5 build: same model, same flags,
# binaries snapshotted from build/bin into ~/inkling-live-bin-p5. Not run by the phase-5 agent; run it yourself.
# Back: systemctl --user stop inkling-3090-experiment && bash perf/live/restore-live.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
UNIT=inkling-3090-experiment
BIN=$HOME/inkling-live-bin-p5
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
envargs=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in LD_LIBRARY_PATH=*) line="LD_LIBRARY_PATH=$BIN";; esac
  envargs+=("--setenv=$line")
done < "$D/live-env.txt"
# shellcheck disable=SC2046
systemd-run --user --unit="$UNIT" --description="Inkling 2 Mini single-3090 OpenCode experiment (phase 5 build)" \
  --working-directory="$R" "${envargs[@]}" $(sed "s#/home/arelius/inkling-live-bin/#$BIN/#" "$D/live-cmd.txt")
for i in $(seq 1 120); do curl -sf -m2 http://127.0.0.1:8031/health >/dev/null && { echo "live server on :8031 from $BIN"; exit 0; }; sleep 2; done
echo "live server did not become healthy; check: journalctl --user -u $UNIT -n 50"; exit 1
