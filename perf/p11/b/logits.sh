#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
mkdir -p "$out"
for depth in 200 4096 16384 32768 65536; do
    dir="build/p11/logits-$depth"
    mkdir -p "$dir"
    for leg in p9 new; do
        libs="$HOME/inkling-live-bin-p9-72fd097"
        if [[ $leg == new ]]; then libs=${BRACKET_NEW:?candidate runtime directory}; fi
        B_LIBS="$libs" perf/p11/b/gpu.sh perf/p11/b/run.sh logits "$depth" 8 >"$dir/$leg.f32" 2>"$dir/$leg.log"
        test "$(stat -c %s "$dir/$leg.f32")" -eq "$((8 * 200058 * 4))"
    done
    sha256sum "$dir/p9.f32" "$dir/new.f32" >"$out/$depth.sha256"
    cmp "$dir/p9.f32" "$dir/new.f32"
done
