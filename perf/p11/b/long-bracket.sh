#!/usr/bin/env bash
set -euo pipefail
ctx=${1:?slot size}
depth=${2:?prompt tokens}
name=${3:?label}
plain=${4:?verified plain capture label}
for leg in old1 new1 new2 old2; do
    export B_LIBS="$HOME/inkling-live-bin-p9-72fd097"
    cap=6
    export B_UBATCH=512 B_MMA=3 B_WINDOW_COLS=1
    if [[ $leg == new* ]]; then B_LIBS=${BRACKET_NEW:?candidate runtime}; cap=${BRACKET_CAP:-6}; B_UBATCH=${BRACKET_NEW_UB:-512}; B_MMA=${BRACKET_NEW_MMA:-3}; B_WINDOW_COLS=${BRACKET_NEW_WINDOW:-1}; fi
    if [[ ${B_RESUME:-0} != 1 || ! -s perf/p11/b/long/$name-$leg.equal ]]; then
        perf/p11/b/gpu.sh perf/p11/b/long.sh "$ctx" "$depth" "$cap" "$name-$leg"
    fi
    python3 - "$depth" "$plain" "$name-$leg" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path

depth, ref, label = sys.argv[1:]
root = Path('build/p11/long')
base = json.loads((root / f'{ref}-plain-edit-{depth}.json').read_text())
data = json.loads((root / f'{label}-dspark-edit-{depth}.json').read_text())
assert base['message'] == data['message'] and base['ids'] == data['ids']
assert len(data['ids']) == 3000 and int(depth) == data['timings']['prompt_n']
digest = hashlib.sha256(struct.pack('<3000i', *data['ids'])).hexdigest()
Path(f'perf/p11/b/long/{label}.equal').write_text(f'Full message and 3000 IDs match {ref} plain. IDs SHA256 {digest}\n')
PY
done
