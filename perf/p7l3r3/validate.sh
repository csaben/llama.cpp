#!/usr/bin/env bash
# Run from the worktree root after a Release build. The archived baseline is built separately.
set -euo pipefail
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 LD_LIBRARY_PATH="$PWD/build/bin"
work=${P7L3R3_WORK:-build/p7l3r3}
out=${P7L3R3_OUT:-perf/p7l3r3}
mkdir -p "$work" "$out"
if [[ ${P7L3R3_DEFAULT:-0} == 1 ]]; then
    unset IK_DEPENDENCY_FUSION
else
    export IK_DEPENDENCY_FUSION=1
fi
base="$PWD/build/p7l3r2/baseline-16264c6/obj/bin"
common=(-m "$Q2" -ngl 99 -ctk q8_0 -ctv q8_0)
bench=(-fa 1 -p 0 -n 128 -r 3)
ppl=(-fa on -f perf/ppl.txt)
cool() {
    local temp
    while :; do
        temp=$(nvidia-smi -i 00000000:01:00.0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        if ((temp <= 63)); then break; fi
        sleep 3
    done
    echo "start_temp_C=$temp"
}
bench_one() (
    local version=$1 depth=$2 bin="$PWD/build/bin"
    if [[ $version == base ]]; then bin=$base; fi
    if [[ $version == off ]]; then export IK_DEPENDENCY_FUSION=0; fi
    echo "version=$version depth=$depth"
    cool
    LD_LIBRARY_PATH="$bin" "$bin/llama-bench" "${common[@]}" "${bench[@]}" -d "$depth"
)
case "$1" in
    bench)
        for version in base on; do
            bench_one "$version" 0,4096,16384,32768 >"$out/official-$version.txt" 2>"$work/official-$version.log"
        done
        for depth in 0 4096 16384 32768; do
            for version in off on on off; do bench_one "$version" "$depth"; done
        done >"$out/paired.txt" 2>"$work/paired.log"
        ;;
    quality)
        g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p7l3/logits.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o "$work/logits"
        for version in base on; do
            bin="$PWD/build/bin"
            if [[ $version == base ]]; then bin=$base; fi
            for workload in prefill decode; do
                args=(-c 2048 -b 2048 --chunks 8)
                if [[ $workload == decode ]]; then args=(-c 1024 -b 1024 -ub 1 --chunks 2); fi
                LD_LIBRARY_PATH="$bin" "$bin/llama-perplexity" "${common[@]}" "${ppl[@]}" "${args[@]}" >"$work/$version-$workload.log" 2>&1
            done
            for prefix in 200 4096 16384 32768; do
                rows=8
                if [[ $prefix == 200 ]]; then rows=64; fi
                LD_LIBRARY_PATH="$bin" "$work/logits" "$Q2" perf/ppl.txt "$prefix" "$rows" >"$work/$version-$prefix.bin" 2>"$work/$version-$prefix.log"
                test "$(stat -c %s "$work/$version-$prefix.bin")" -eq "$((rows*200058*4))"
            done
        done
        rg 'calculating perplexity|Final estimate' "$work"/{base,on}-{prefill,decode}.log >"$out/ppl.txt"
        for prefix in 200 4096 16384 32768; do
            cmp "$work/base-$prefix.bin" "$work/on-$prefix.bin"
            sha256sum "$work"/{base,on}-$prefix.bin
        done >"$out/logits.txt"
        ;;
    mr)
        export IK_MR=1 IK_MMA=3 MR_RS_A=0 MR_RS=4
        g++ -O2 -std=c++17 -Wno-deprecated-declarations perf/p6/mr_check.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o "$work/mr_check"
        gzip -dc perf/p6/r4_repro/n3.trace.gz >"$work/r4.trace"
        (
            MR_RS=3 MR_CTX=16384 MR_TRACE="$PWD/$work/r4.trace" "$work/mr_check" "$Q2" perf/ppl.txt
            "$work/mr_check" "$Q2" perf/ppl.txt 300 24
            for prefix in 1533 3069 4500; do
                MR_RAND=2 MR_SIZES=5 MR_PLAIN_PCT=0 "$work/mr_check" "$Q2" perf/ppl.txt "$prefix" 24
            done
            for prefix in 2000 2046 2558 3069 4094 4500; do
                MR_RAND=37 MR_SIZES=2,3,4,5 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 "$work/mr_check" "$Q2" perf/ppl.txt "$prefix" 64
            done
            IK_ATTN_FIXED_MR=0 MR_RAND=2 MR_SIZES=5 "$work/mr_check" "$Q2" perf/ppl.txt 3069 24
        ) >"$work/mr.log" 2>&1
        rg 'rows differ' "$work/mr.log" >"$out/mr.txt"
        test "$(wc -l < "$out/mr.txt")" -eq 12
        if rg -F -v ': 0 rows differ from single-token decode, max |logit diff| 0' "$out/mr.txt"; then exit 1; fi
        ;;
    ops)
        for ops in MUL_MAT,MUL_MAT_ID INKLING_ATTN,INKLING_ROUTER,SHORTCONV_RES,RES_MIX,LOOP_CTRL_DOWN,LOOP_CTRL_APPLY SITU CONCAT; do
            args=()
            # The expert sum and shortconv fixtures concatenate the state and output.
            if [[ $ops == CONCAT ]]; then args=(-p 'n_slots=|n_used='); fi
            build/bin/test-backend-ops test -b CUDA0 -o "$ops" "${args[@]}" >"$work/ops-$ops.log" 2>&1
            echo "$ops"
            rg 'tests passed|Backend CUDA0.*OK' "$work/ops-$ops.log" | sed -E 's/\x1B\[[0-9;]*[mK]//g'
        done >"$out/ops.txt"
        test "$(rg -c 'Backend CUDA0: OK' "$out/ops.txt")" -eq 4
        ;;
    guards)
        for cfg in GGML_CUDA_DISABLE_GRAPHS=1 IK_DENSE_VISIT=0 IK_MR=1 IK_CTRL_APPLY8=0 IK_MIX_SPECIALIZE=0 IK_PROJ_CONV=0; do
            for enabled in 0 1; do
                env "$cfg" IK_DEPENDENCY_FUSION=$enabled "$work/logits" "$Q2" perf/ppl.txt 200 8 >"$work/guard-$cfg-$enabled.bin" 2>"$work/guard-$cfg-$enabled.log"
                test "$(stat -c %s "$work/guard-$cfg-$enabled.bin")" -eq "$((8*200058*4))"
            done
            cmp "$work/guard-$cfg-0.bin" "$work/guard-$cfg-1.bin"
            echo "$cfg 8 full vocabulary rows identical"
        done >"$out/fallbacks.txt"
        ;;
    *) exit 2 ;;
esac
