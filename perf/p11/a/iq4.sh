#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export P11_MODEL="$HOME/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-iq4nl.gguf"
export P11_CTX=8192 P11_UB=128 P11_DRAFT_UB=128 SPEC_REPS=3 SPEC_MAXT=384
export P11_OUT="$PWD/perf/p11/a/iq4" P11_TELEMETRY=1
mkdir -p "$P11_OUT"
perf/p11/a/probe.sh plain one plain
for leg in olda newa newb oldb; do
    (
        dev=one
        if [[ $leg == new* ]]; then
            dev=head
            export P11_BIN="${1:?runtime}" IK_PEER_NATIVE=1 IK_PEER_DIRECT=1
        fi
        perf/p11/a/probe.sh bench "$dev" "iq4-bench-$leg" -p 0 -n 128 -r 5
        perf/p11/a/probe.sh draft "$dev" "iq4-edit-$leg"
        SPEC_KINDS=edit SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$P11_OUT" plain "iq4-edit-$leg" >"$P11_OUT/$leg.equal"
    )
done
