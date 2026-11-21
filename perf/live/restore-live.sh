#!/bin/bash
# Restore the user's live Inkling endpoint (localhost:8031, 2-bit model, 64K ctx) exactly as it ran before the
# phase-5 round paused it. Safe to run twice: it does nothing if the unit is already active.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
UNIT=inkling-3090-experiment
if systemctl --user is-active --quiet "$UNIT"; then echo "$UNIT already active"; exit 0; fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
envargs=()
while IFS= read -r line; do [ -n "$line" ] && envargs+=("--setenv=$line"); done < "$D/live-env.txt"
# shellcheck disable=SC2046
systemd-run --user --unit="$UNIT" --description="Inkling 2 Mini single-3090 OpenCode experiment" \
  --working-directory=/home/arelius/llama.cpp-fast "${envargs[@]}" $(cat "$D/live-cmd.txt")
for i in $(seq 1 120); do curl -sf -m2 http://127.0.0.1:8031/health >/dev/null && { echo "live server back on :8031"; exit 0; }; sleep 2; done
echo "live server did not become healthy; check: journalctl --user -u $UNIT -n 50"; exit 1
