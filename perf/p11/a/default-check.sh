#!/usr/bin/env bash
# Compare the default one-card build with p9, including two interleaved sequences.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?label} runtime=${2:?runtime}
unset IK_PEER_HEAD IK_PEER_NATIVE IK_PEER_EXPERTS IK_PEER_FLAGS IK_PEER_DIRECT IK_PEER_PREFIX IK_PEER_GATES IK_PEER_HEAVY
for mode in 0 1; do
    export P11_MR=$mode
    for version in p9 "$name"; do
        export P11_BIN="$HOME/inkling-live-bin-p9-72fd097"
        if [[ $version != p9 ]]; then export P11_BIN="$runtime"; fi
        label="$version-default-mixed-$mode"
        perf/p11/a/check.sh one logits "$label" 1129 32 interleave-mixed
        test "$(stat -c %s "build/p11a/checks/$label.log.bin")" -eq $((64*200058*4))
    done
    cmp "build/p11a/checks/p9-default-mixed-$mode.log.bin" "build/p11a/checks/$name-default-mixed-$mode.log.bin"
done
