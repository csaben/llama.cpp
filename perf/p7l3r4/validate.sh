#!/usr/bin/env bash
# Run from the worktree root. Each invocation owns GPU0 for its duration.
set -euo pipefail
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 LD_LIBRARY_PATH="$PWD/build/bin"
export P7L3R3_DEFAULT=1 P7L3R3_WORK=build/p7l3r4/final P7L3R3_OUT=perf/p7l3r4/final
work=$P7L3R3_WORK out=$P7L3R3_OUT
mkdir -p "$work" "$out"
unset IK_GLUE_FAST IK_DEPENDENCY_FUSION
exec 9>build/p7l3r4/gpu0.lock
flock 9
case "$1" in
    quality|mr|ops)
        bash perf/p7l3r3/validate.sh "$1"
        ;;
    guards)
        for config in GGML_CUDA_DISABLE_GRAPHS=1 IK_DEPENDENCY_FUSION=0 IK_MR=1 IK_DENSE_VISIT=0 IK_CTRL_APPLY8=0 IK_MIX_SPECIALIZE=0 IK_PROJ_CONV=0 IK_ATTN_SHORT=0; do
            for value in 0 1; do
                env "$config" IK_GLUE_FAST=$value "$work/logits" "$Q2" perf/ppl.txt 200 8 >"$work/guard-$config-$value.bin" 2>"$work/guard-$config-$value.log"
                test "$(stat -c %s "$work/guard-$config-$value.bin")" -eq "$((8*200058*4))"
            done
            cmp "$work/guard-$config-0.bin" "$work/guard-$config-1.bin"
            echo "$config 8 full-vocabulary rows identical"
        done >"$out/fallbacks.txt"
        ;;
    sequences)
        base="$PWD/build/p7l3r2/baseline-16264c6/obj/bin"
        for prefix in 49 1129; do
            for mode in base off on; do
                bin="$PWD/build/bin"
                value=1
                if [[ $mode == base ]]; then bin=$base; fi
                if [[ $mode == off ]]; then value=0; fi
                LD_LIBRARY_PATH="$bin" IK_GLUE_FAST=$value "$work/logits" "$Q2" perf/ppl.txt "$prefix" 32 interleave >"$work/seq-$prefix-$mode.bin" 2>"$work/seq-$prefix-$mode.log"
                test "$(stat -c %s "$work/seq-$prefix-$mode.bin")" -eq "$((64*200058*4))"
            done
            cmp "$work/seq-$prefix-base.bin" "$work/seq-$prefix-off.bin"
            cmp "$work/seq-$prefix-base.bin" "$work/seq-$prefix-on.bin"
            sha256sum "$work"/seq-"$prefix"-{base,off,on}.bin
        done >"$out/sequences.txt"
        ;;
    bench)
        base="$PWD/build/p7l3r2/baseline-16264c6/obj/bin"
        bash perf/p7l3r4/run.sh bench final/official-base 0,4096,16384,32768 "$base"
        bash perf/p7l3r4/run.sh bench final/official-on 0,4096,16384,32768
        for depth in 0 4096 16384 32768; do
            R4_MAX_TEMP=63 bash perf/p7l3r4/pair.sh "final/paired-$depth" "$depth" IK_GLUE_FAST 1
        done
        ;;
    trace)
        for depth in 0 16384; do
            IK_GLUE_FAST=0 bash perf/p7l3r4/run.sh trace "final/off-$depth" "$depth"
            bash perf/p7l3r4/run.sh trace "final/on-$depth" "$depth"
        done
        python3 perf/p7l3r4/summarize.py "$work"/{off,on}-{0,16384}.sqlite >"$out/coverage.txt"
        for depth in 0 16384; do
            python3 perf/p7l3r4/compare.py "$work/off-$depth.sqlite" "$work/on-$depth.sqlite" "$depth"
        done >"$out/kernel-deltas.md"
        for depth in 0 16384; do
            python3 perf/p7l3r4/latency.py "$work/on-$depth.sqlite" "$depth" --mean
        done >"$out/latency-mean.md"
        ;;
    *) exit 2 ;;
esac
