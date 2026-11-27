#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
mkdir -p "$out"
for depth in 200 4096 16384 32768 65536; do
    dir="${B_LOGITS_WORK:-build/p11}/logits-$depth"
    mkdir -p "$dir"
    for leg in p9 new; do
        libs="$HOME/inkling-live-bin-p9-72fd097"
        export B_MMA=3 B_WINDOW_COLS=1
        if [[ $leg == new ]]; then
            libs=${BRACKET_NEW:?candidate runtime directory}
            B_MMA=${BRACKET_NEW_MMA:-3}
            B_WINDOW_COLS=${BRACKET_NEW_WINDOW:-1}
        fi
        if [[ ${B_RESUME:-0} != 1 || $(stat -c %s "$dir/$leg.f32" 2>/dev/null || :) != "$((8 * 200058 * 4))" ]]; then
            B_LIBS="$libs" perf/p11/b/gpu.sh perf/p11/b/run.sh logits "$depth" 8 >"$dir/$leg.f32" 2>"$dir/$leg.log"
        fi
        test "$(stat -c %s "$dir/$leg.f32")" -eq "$((8 * 200058 * 4))"
    done
    cmp "$dir/p9.f32" "$dir/new.f32"
    sha256sum "$dir/p9.f32" "$dir/new.f32" >"$out/$depth.sha256"
done
