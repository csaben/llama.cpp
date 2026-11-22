#!/usr/bin/env bash
# Run each stage serially on the release card.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 CHECKS_GPU=1 P8_GPU=1
export VERIFY_WORK=build/p9/verify
out=perf/p9
mkdir -p "$VERIFY_WORK"
case ${1:?matrix, snapshot or a Phase 8 merge verification stage} in
    matrix)
        dir=${2:-$out/matrix}
        bash perf/p8bc/matrix.sh "$dir"
        python3 perf/p8bc/summarize.py "$dir" >"$dir/cells.tsv"
        python3 perf/p8m/audit.py "$dir" perf/p8m/matrix/cells.tsv >"$dir/audit.txt" ;;
    interleave)
        source perf/p5env.sh
        g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p7l3/logits.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o "$VERIFY_WORK/logits"
        for mode in default mr; do
            flags=()
            if [[ $mode == mr ]]; then flags=(IK_MR=1 IK_MMA=3); fi
            for version in frozen candidate; do
                libs=$HOME/inkling-live-bin-p8m-f7f505c
                if [[ $version == candidate ]]; then libs="$PWD/build/bin"; fi
                file="$VERIFY_WORK/interleave-$mode-$version.bin"
                env "${flags[@]}" LD_LIBRARY_PATH="$libs" "$VERIFY_WORK/logits" "$Q2" perf/ppl.txt 1129 32 interleave-mixed >"$file" 2>"$file.log"
                test "$(stat -c %s "$file")" -eq "$((64*200058*4))"
                sha256sum "$file"
            done
            cmp "$VERIFY_WORK/interleave-$mode-frozen.bin" "$VERIFY_WORK/interleave-$mode-candidate.bin"
        done >"$out/checks/interleave-mixed.txt" ;;
    tiles)
        args=(build/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'rel_bias=1')
        for mode in test memcheck synccheck; do
            prefix=()
            log="$VERIFY_WORK/direct-$mode"
            if [[ $mode != test ]]; then
                prefix=("$HOME/opt/cuda-12.8/bin/compute-sanitizer" --tool "$mode" --error-exitcode 86 --log-file "$log.sanitizer")
            fi
            LD_LIBRARY_PATH="$PWD/build/bin" "${prefix[@]}" "${args[@]}" >"$log.ops" 2>&1
            echo "$mode"
            rg ' 7/7 tests passed' "$log.ops"
            if [[ $mode != test ]]; then rg 'ERROR SUMMARY: 0 errors' "$log.sanitizer"; fi
        done >"$out/checks/q8-tiles.txt" ;;
    snapshot)
        bash perf/p8m/snapshot.sh "${2:?snapshot directory}" 6 "$out/snapshot-probes" "$out/matrix/main/384" ;;
    *) bash perf/p8m/checks.sh "$1" "$out/checks" ;;
esac
