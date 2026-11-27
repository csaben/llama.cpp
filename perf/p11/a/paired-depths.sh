#!/usr/bin/env bash
# Each lease runs one llama-bench invocation with the depth list in the same order.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?label} candidate=${2:?runtime} mode=${3:?tg or pp}
export P11_OUT="$PWD/perf/p11/a/paired/$name" P11_TELEMETRY=1
mkdir -p "$P11_OUT"
case $mode in
    tg) args=(-p 0 -n 128 -d 0,4096,16384,32768,65536);;
    pp) args=(-p 4096,16384,65536 -n 0);;
    *) exit 2;;
esac
for leg in olda newa newb oldb; do
    (
        unset IK_PEER_HEAD IK_PEER_NATIVE IK_PEER_EXPERTS IK_PEER_FLAGS IK_PEER_DIRECT
        export P11_BIN="$HOME/inkling-live-bin-p9-72fd097"
        placement=one
        if [[ $leg == new* ]]; then
            export P11_BIN="$candidate" IK_PEER_NATIVE=1 IK_PEER_DIRECT=1
            placement=head
        fi
        label="$name-$leg-$mode"
        if [[ ! -f "$P11_OUT/$label.done" ]]; then
            perf/p11/a/probe.sh bench "$placement" "$label" "${args[@]}" -r 3
            python3 - "$P11_OUT/$label.json" "$mode" <<'PY'
import json
import sys
from pathlib import Path

path, mode = Path(sys.argv[1]), sys.argv[2]
rows = json.loads(path.read_text())
field = "n_depth" if mode == "tg" else "n_prompt"
assert [row[field] for row in rows] == ([0, 4096, 16384, 32768, 65536] if mode == "tg" else [4096, 16384, 65536])
for row in rows:
    assert len(row["samples_ts"]) == 3 and min(row["samples_ts"]) > 0
PY
            touch "$P11_OUT/$label.done"
        fi
    )
done
