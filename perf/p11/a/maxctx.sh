#!/usr/bin/env bash
# Measure completed near-full vision requests on the active memory configuration.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?label} place=${2:?one or head} runtime=${3:?runtime}
shift 3
export P11_BIN="$runtime"
if [[ $place == head ]]; then export IK_PEER_NATIVE=1 IK_PEER_DIRECT=1; fi
for ctx in "$@"; do
    depth=$((ctx-3008))
    label="$name-$ctx"
    if perf/p11/a/capacity.sh "$place" draft "$label" "$ctx" "$depth" 3000; then
        ref="p9-ref-$ctx"
        if [[ ! -s perf/p11/a/capacity/$ref.ids ]]; then
            env -u IK_PEER_NATIVE -u IK_PEER_DIRECT -u IK_PEER_HEAD P11_BIN="$HOME/inkling-live-bin-p9-72fd097" perf/p11/a/capacity.sh one plain "$ref" "$ctx" "$depth" 3000
        fi
        for ext in ids json; do cmp "perf/p11/a/capacity/$ref.$ext" "perf/p11/a/capacity/$label.$ext"; done
        echo "$ctx completed and matched"
        break
    else
        rc=$?
        [[ $rc != 75 ]] || exit "$rc"
        rg -i 'out of memory|failed to allocate|failed to reserve' "build/p11a/capacity/$label.server" >"perf/p11/a/capacity/$label.oom" || exit "$rc"
        echo "$ctx exceeded memory"
    fi
done
