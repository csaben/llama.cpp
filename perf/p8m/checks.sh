#!/usr/bin/env bash
# Run the Phase 8a battery and the extra verifier widths and CUDA cases from B/C.
set -euo pipefail
source perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CHECKS_GPU=1 LD_LIBRARY_PATH="$PWD/build/bin"
export VERIFY_WORK=build/p8m VERIFY_CORE_OPS=1110 VERIFY_MATRIX_OPS=3779
out=perf/p8m/checks
mkdir -p "$out" "$VERIFY_WORK"
case "${1:?stage}" in
    checks|checks-default|depth|logits|ops|sync)
        bash perf/p8/verify.sh "$1" "$out" ;;
    wide)
        cat perf/ppl.txt perf/ppl.txt >"$VERIFY_WORK/depth-input"
        for packed in 0 1; do
            for depth in 200 4096 16384 32768; do
                log="$VERIFY_WORK/wide-$packed-$depth.log"
                IK_MR=1 IK_MMA=3 IK_WINDOW_COLS=$packed MR_RS_A=0 MR_RS=6 MR_SIZES=2,3,4,5,6,7 MR_RAND=37 MR_PLAIN_PCT=0 MR_CTX=34816 \
                    build/bin/mr_check "$Q2" "$VERIFY_WORK/depth-input" "$depth" 64 >"$log" 2>&1
                echo "packed=$packed $(rg '182 rows .*: 0 rows differ from single-token decode, max \|logit diff\| 0$' "$log")"
                rg -q '182 rows .*: 0 rows differ from single-token decode, max \|logit diff\| 0$' "$log"
            done
        done >"$out/wide.txt" ;;
    extra-ops)
        for op in REL_BIAS CONCAT; do
            build/bin/test-backend-ops test -b CUDA0 -o "$op" >"$VERIFY_WORK/extra-$op.log" 2>&1
            rg 'tests passed' "$VERIFY_WORK/extra-$op.log"
        done >"$out/extra-ops.txt"
        rg -q '13/13 tests passed' "$out/extra-ops.txt"
        rg -q '273/273 tests passed' "$out/extra-ops.txt"
        for op in MUL_MAT_ID REL_BIAS; do
            filter='m=257,n=512,k=512'
            count=2
            if [[ $op == REL_BIAS ]]; then filter='n_tok=129,n_kv=1025'; count=1; fi
            for mode in memcheck synccheck; do
                log="$VERIFY_WORK/$mode-$op.log"
                "$HOME/opt/cuda-12.8/bin/compute-sanitizer" --tool "$mode" --error-exitcode 86 --log-file "$log" \
                    build/bin/test-backend-ops test -b CUDA0 -o "$op" -p "$filter" >"$log.ops" 2>&1
                echo "$mode $op"
                rg " $count/$count tests passed" "$log.ops"
                rg 'ERROR SUMMARY: 0 errors' "$log"
            done
        done >"$out/extra-sanitizer.txt" ;;
    *) exit 2 ;;
esac
