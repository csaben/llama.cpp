#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?label} candidate=${2:?runtime directory} suite=${3:?ppl, rollback, logits, or ops}
export P11_BIN="$candidate" IK_PEER_HEAD=CUDA1 IK_PEER_NATIVE=1
check=perf/p11/a/check.sh
q2="$HOME/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf"
case $suite in
    ppl)
        for variant in uncensored calibrated; do
            if [[ $variant == calibrated ]]; then export P11_MODEL="$q2"; fi
            "$check" peer prefill "$name-$variant-pp"
            "$check" peer decode "$name-$variant-tg"
            P11_MR=1 "$check" peer decode "$name-$variant-tg-mr"
        done;;
    rollback)
        export P11_MR=1 MR_RS_A=0 MR_RS=4
        mkdir -p build/p11a/checks
        gzip -dc perf/p6/r4_repro/n3.trace.gz >"build/p11a/checks/$name.trace"
        MR_RS=3 MR_CTX=16384 MR_TRACE="$PWD/build/p11a/checks/$name.trace" "$check" peer mr "$name-trace"
        for prefix in 1533 3069 4500; do
            MR_RAND=2 MR_SIZES=5 MR_PLAIN_PCT=0 "$check" peer mr "$name-edge-$prefix" "$prefix" 24
        done
        IK_ATTN_FIXED_MR=0 MR_RAND=2 MR_SIZES=5 "$check" peer mr "$name-control" 3069 24
        for prefix in 2000 2046 2558 3069 4094 4500; do
            MR_RAND=37 MR_SIZES=2,3,4,5 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 "$check" peer mr "$name-rand-$prefix" "$prefix" 64
        done
        MR_CTX=18432 MR_RS=6 MR_RAND=37 MR_SIZES=2,3,4,5,6,7 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 "$check" peer mr "$name-widths" 16384 96;;
    logits)
        for mode in 0 1; do
            for depth in 1 4096 16384 32768 65528; do
                label="logits-$mode-$depth"
                if [[ ! -s build/p11a/checks/p9-$label.log.bin ]]; then
                    env -u IK_PEER_HEAD -u IK_PEER_NATIVE P11_BIN="$HOME/inkling-live-bin-p9-72fd097" P11_MR="$mode" "$check" one logits "p9-$label" "$depth" 8
                fi
                P11_MR="$mode" "$check" peer logits "$name-$label" "$depth" 8
                cmp "build/p11a/checks/p9-$label.log.bin" "build/p11a/checks/$name-$label.log.bin"
            done
        done;;
    ops)
        "$check" peer ops "$name-ops-head" -o MUL_MAT -p 'type_a=iq4_nl|type=iq4_nl'
        "$check" peer ops "$name-ops-inkling" -o MUL_MAT_ID,INKLING_ROUTER,INKLING_ATTN;;
    *) exit 2;;
esac
