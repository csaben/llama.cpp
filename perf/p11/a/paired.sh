#!/usr/bin/env bash
# Each placement runs one benchmark per lease. The two outer legs use the frozen p9 runtime.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?label} candidate=${2:?runtime directory} suite=${3:?bench, draft, or capacity}
out="$PWD/perf/p11/a/paired/$name"
mkdir -p "$out" build/p11a
budget_filter=${P11_PAIR_BUDGET:-}
if [[ -n $budget_filter ]]; then
    [[ $suite == draft && ( $budget_filter == 384 || $budget_filter == 3000 ) ]] || exit 2
fi
exec {suite_lock}>"build/p11a/paired-$name-$suite${budget_filter:+-$budget_filter}.lock"
flock "$suite_lock"
for leg in olda newa newb oldb; do
    (
        unset IK_PEER_HEAD IK_PEER_NATIVE IK_PEER_EXPERTS IK_PEER_FLAGS IK_PEER_DIRECT
        export P11_BIN="$HOME/inkling-live-bin-p9-72fd097" P11_OUT="$out" P11_ADAPT=1 P11_WIDTH=6
        placement=one
        if [[ $leg == new* ]]; then
            export P11_BIN="$candidate" IK_PEER_HEAD=CUDA1 IK_PEER_NATIVE=1
            placement=head
            if [[ ${P11_DIRECT:-0} == 1 ]]; then export IK_PEER_DIRECT=1; fi
        fi
        run() {
            local tag=$1 mode=$2
            shift 2
            if [[ ! -f "$out/$leg-$tag.done" ]]; then
                perf/p11/a/probe.sh "$mode" "$placement" "$name-$leg-$tag" "$@"
                touch "$out/$leg-$tag.done"
            fi
        }
        case $suite in
            bench)
                for depth in 0 4096 16384 32768 65536; do
                    run "tg$depth" bench -p 0 -n 128 -d "$depth" -r 3
                done
                for depth in 4096 16384 65536; do
                    run "pp$depth" bench -p "$depth" -n 0 -r 3
                done;;
            draft)
                export SPEC_REPS=3 P11_KINDS=edit
                for budget in 384 3000; do
                    if [[ -n $budget_filter && $budget != "$budget_filter" ]]; then continue; fi
                    export SPEC_MAXT="$budget"
                    run "edit$budget" draft
                done;;
            capacity)
                tag="$name-$leg-128"
                if [[ ! -f "$out/$leg-128.done" ]]; then
                    perf/p11/a/capacity.sh "$placement" draft "$tag" 131072 128064 3000
                    for ext in ids json; do cmp "perf/p11/a/capacity/p9-128-plain.$ext" "perf/p11/a/capacity/$tag.$ext"; done
                    touch "$out/$leg-128.done"
                fi;;
            *) exit 2;;
        esac
    )
done
