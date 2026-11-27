#!/usr/bin/env bash
# Replay one previously contended matrix cell and check its output against the independent p9 controls.
set -euo pipefail
cd "$(dirname "$0")/../../.."
family=$1 suite=$2 budget=$3 setting=$4
case $family in
    release) libs=$PWD/build/p11a/repro-direct/bin;;
    direct|direct64) libs=$PWD/build/p11a/direct-head;;
    native) libs=$PWD/build/p11a/native-head;;
    p9|p964) libs=$HOME/inkling-live-bin-p9-72fd097;;
    *) exit 2;;
esac
export P11_BIN=$libs P11_CTX=131072 P11_DRAFT_UB=512 P11_POLICY=1
export SPEC_MAXT="$budget" SPEC_REPS=3 P11_KINDS=code,edit,prose,edit@16K,edit@32K,agent
if [[ $suite == prose ]]; then export SPEC_REPS=5 P11_KINDS=letter,science,scene,review,dialogue; fi
if [[ $suite == fixed ]]; then export P11_POLICY=0; fi
if [[ $budget == 3000 ]]; then export SPEC_REPS=1; fi
dev=head
if [[ $family == p9* ]]; then dev=one; else export IK_PEER_NATIVE=1; fi
if [[ $family == direct* || $family == release ]]; then export IK_PEER_DIRECT=1; fi
if [[ $family == *64 ]]; then export P11_CTX=65536; fi
mode=draft
if [[ $setting == plain ]]; then mode=plain; fi
export P11_ADAPT=0 P11_WIDTH=${setting:1:1}
if [[ $setting == *-adapt ]]; then export P11_ADAPT=1; fi
d="perf/p11/a/replay/$family/$suite/$budget"
log="build/p11a/replay/$family/$suite/$budget"
mkdir -p "$d" "$log"
if [[ ! -e "$log/$setting.done" ]]; then
    P11_OUT="$d" P11_LOG="$log/$setting.server" perf/p11/a/probe.sh "$mode" "$dev" "$setting"
    for kind in ${P11_KINDS//,/ }; do
        for ext in ids json; do
            cmp "$d/$setting.$kind.$ext" "perf/p11/a/p9-matrix/$suite/$budget/plain.$kind.$ext"
        done
    done
    touch "$log/$setting.done"
fi
