#!/usr/bin/env bash
# Load the staged runtime on the lab ports and compare both text and image requests.
set -euo pipefail
cd "$(dirname "$0")/../../.."
read -r source mode draft_ub <perf/p11/a/release.txt
export P11_BIN="$HOME/inkling-live-bin-p11a-$source" P11_DRAFT_UB="$draft_ub" P11_CTX=131072
manifest=$PWD/perf/p11/a/snapshot.sha256
(cd "$P11_BIN" && cmp SHA256SUMS "$manifest" && sha256sum --strict -c "$manifest")
placement=one
if [[ $mode == head || $mode == direct ]]; then
    placement=head
    export IK_PEER_NATIVE=1
    if [[ $mode == direct ]]; then export IK_PEER_DIRECT=1; fi
fi
out=$PWD/perf/p11/a/snapshot-probes
mkdir -p "$out"
P11_OUT="$out" P11_KINDS=edit,agent P11_WIDTH=6 P11_ADAPT=1 SPEC_REPS=3 SPEC_MAXT=384 perf/p11/a/probe.sh draft "$placement" staged
for kind in edit agent; do
    for ext in ids json; do cmp "$out/staged.$kind.$ext" "perf/p11/a/p9-matrix/main/384/plain.$kind.$ext"; done
done
P11_IMAGE=tools/mtmd/test-1.jpeg perf/p11/a/capacity.sh "$placement" draft snapshot-image 131072 0 384
for ext in ids json; do cmp "perf/p11/a/capacity/snapshot-image.$ext" "perf/p11/a/capacity/p9-real-image.$ext"; done
python3 - <<'PY'
import json
from pathlib import Path
p = Path("perf/p11/a/capacity/snapshot-image.timings")
t = json.loads(p.read_text())
assert t["cache_n"] == 0 and t["prompt_n"] == 238 and t["predicted_n"] == 384
PY
echo "Six staged text replies and one staged JPEG reply matched the p9 plain controls on every message byte and all 2,688 generated IDs."
