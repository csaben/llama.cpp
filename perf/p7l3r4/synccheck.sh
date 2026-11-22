#!/usr/bin/env bash
# Check the short-scan named barrier and the cooperative controller pack on real decode steps.
set -euo pipefail
. perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU=0 LD_LIBRARY_PATH="$PWD/build/bin"
export GGML_CUDA_DISABLE_GRAPHS=1
unset IK_GLUE_FAST IK_DEPENDENCY_FUSION
work=build/p7l3r4/final
out=perf/p7l3r4/final
exec 9>build/p7l3r4/gpu0.lock
flock 9
for prefix in 113 1129; do
    "$HOME/opt/cuda-12.8/bin/compute-sanitizer" --tool synccheck --error-exitcode 86 \
        --kernel-name kns=ik_dense_gate_scan --kernel-name kns=ik_dense_warp_roles --kernel-name kns=ik_ctrl_apply_pack \
        --log-file "$work/synccheck-$prefix.log" "$work/logits" "$Q2" perf/ppl.txt "$prefix" 2 >"$work/synccheck-$prefix.bin" 2>"$work/synccheck-model-$prefix.log"
    test "$(stat -c %s "$work/synccheck-$prefix.bin")" -eq "$((2*200058*4))"
    printf 'prefix %s, two decode steps\n' "$prefix"
    rg 'ERROR SUMMARY: 0 errors' "$work/synccheck-$prefix.log"
done >"$out/synccheck.txt"
