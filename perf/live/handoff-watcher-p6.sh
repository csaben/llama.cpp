#!/bin/bash
# Phase-5 hand-off watcher. Runs outside Claude (systemd --user), so it keeps working if Claude is rate-limited.
#  - Claude usage (from ~/.claude/usage-meter.json, written by the status line) at >= 90% of the 5-hour window or
#    >= 93% of the 7-day window: create perf/STOP so the Fable agent wraps up (leaves ~5% for the user's chats).
#  - The Fable transcript shows a usage-limit error: same as above.
#  - After a stop, once the transcript has been quiet for 5 minutes, start Codex (GPT-6.2 Astra, high) once to continue.
# Usage: handoff-watcher.sh <fable-transcript.jsonl>
set -u
T="$1"; R=/home/arelius/llama.cpp-fast; L="$R/perf/live/handoff-watcher.log"
METER=/home/arelius/.claude/usage-meter.json; FIVE_MAX=90; WEEK_MAX=93
log() { echo "$(date '+%F %T') $*" >> "$L"; }
log "watching $T (5h >= $FIVE_MAX% or 7d >= $WEEK_MAX% -> stop + Codex)"
stopped=0
while true; do
  if [ $stopped -eq 0 ]; then
    read -r five week age < <(python3 - "$METER" <<'EOF' 2>/dev/null || echo "0 0 999999"
import json, os, sys, time
d = json.load(open(sys.argv[1])); r = d.get("rate_limits") or {}
f = (r.get("five_hour") or {}).get("used_percentage", 0); w = (r.get("seven_day") or {}).get("used_percentage", 0)
print(int(f), int(w), int(time.time() - os.path.getmtime(sys.argv[1])))
EOF
)
    reason=""
    [ "$five" -ge $FIVE_MAX ] && reason="5h usage ${five}%"
    [ "$week" -ge $WEEK_MAX ] && reason="7d usage ${week}%"
    tail -c 20000 "$T" 2>/dev/null | grep -q -E "hit your [A-Za-z ]*limit|usage limit|rate_limit" && reason="${reason:-rate-limit error in transcript}"
    if [ -n "$reason" ]; then touch "$R/perf/STOP"; stopped=1; log "STOP: $reason (meter age ${age}s)"; fi
  else
    quiet=$(( $(date +%s) - $(stat -c %Y "$T" 2>/dev/null || echo 0) ))
    if [ $quiet -ge 300 ]; then
      log "Fable quiet ${quiet}s; launching Codex (gpt-6.2-astra, high)"
      rm -f "$R/perf/STOP"
      cd "$R" && codex exec -m gpt-6.2-astra -c model_reasoning_effort=high -C "$R" \
        --dangerously-bypass-approvals-and-sandbox \
        "You are continuing phase 6 of the Inkling 2 Mini decode work in this repo after a Claude agent was stopped to save the user's Claude usage. Read perf/PHASE6_BRIEF.md and follow it exactly (ignore its Claude stop-signal rule), then read perf/HANDOFF.md and perf/RESULTS.md and continue from where the previous agent left off. Keep perf/HANDOFF.md and RESULTS.md updated, commit and tag as the brief says, never push or open PRs, use GPU1 only, and finish with the brief's Finishing steps including bash perf/live/restore-p5.sh." \
        >> "$R/perf/live/codex-phase6.log" 2>&1
      log "Codex exited with $?"; exit 0
    fi
  fi
  sleep 30
done
