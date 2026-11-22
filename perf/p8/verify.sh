#!/usr/bin/env bash
# Run each stage in a fresh shell from the repository root.
set -euo pipefail
source perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH="$PWD/build/bin"
out=${2:-perf/p8/release}
work=${VERIFY_WORK:-build/p8}
frozen="$HOME/inkling-live-bin-p7c-401460d"
mkdir -p "$out" "$work"
release_env() {
    export LLAMA_DSPARK_DEV=1 LLAMA_DSPARK_ROUND=1 LLAMA_DSPARK_GRAPHS=1 LLAMA_DSPARK_POLICY=1
    export IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0
    export LLAMA_ARG_OVERRIDE_TENSOR=token_embd.weight=CUDA0 LLAMA_ARG_SPEC_DRAFT_OVERRIDE_TENSOR=mask_embd.weight=CUDA0
    export SPEC_CTX=65536 SPEC_REQUIRE_IDS=1 SPEC_FIXED_LENGTH=1 MODEL="$Q2"
    export SPEC_KINDS=code,edit,prose,edit@16K,edit@32K
}
check_rows() {
    python3 - "$@" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
rows = re.findall(r'(\d+) (?:output )?rows (?:up to position|in \d+ steps).*?: (\d+) rows differ from single-token decode, max \|logit diff\| (\S+)', text)
assert len(rows) == int(sys.argv[2]) and sum(int(n) for n, _, _ in rows) == int(sys.argv[3]), rows
assert all(int(bad) == 0 and float(error) == 0 for _, bad, error in rows), rows
PY
}
case "${1:?stage}" in
    checks|checks-default)
        mode=$1
        if [[ $mode == checks ]]; then release_env; fi
        CHECKS_OUT="$out/$mode" bash perf/p6/r4_checks.sh
        d="$out/$mode"
        check_rows "$d/rows.txt" 11 2714
        rg -q " ${VERIFY_CORE_OPS:-1108}/${VERIFY_CORE_OPS:-1108} tests passed" "$d/ops.txt"
        for label in plain plain-mr; do
            diff -r "perf/p7c/release/$mode/$label" "$d/$label"
            for ppl in prefill decode; do
                diff <(sed 's/.*Final estimate: //' "perf/p7c/release/$mode/$label.$ppl.txt") <(sed 's/.*Final estimate: //' "$d/$label.$ppl.txt")
            done
        done
        echo 'Six greedy outputs match frozen Phase 7c.' >"$d/greedy.txt" ;;
    depth)
        release_env
        cat perf/ppl.txt perf/ppl.txt >"$work/depth-input"
        for depth in 0 4096 16384 32768; do
            for pack in 0 1; do
                echo "depth=$depth IK_WINDOW_COLS=$pack"
                IK_MR=1 IK_MMA=3 MR_RS_A=0 MR_RS=4 MR_SIZES=4 MR_ROLLBACK=0 MR_CTX=34816 IK_WINDOW_COLS=$pack \
                    build/bin/mr_check "$Q2" "$work/depth-input" "$depth" 32 2>&1 | rg 'window of|rows differ'
            done
        done >"$out/depths.txt"
        check_rows "$out/depths.txt" 8 1024 ;;
    logits)
        g++ -O2 -std=c++17 perf/p7l3/logits.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o "$work/logits"
        for setting in default mr; do
            flags=()
            if [[ $setting == mr ]]; then flags=(IK_MR=1 IK_MMA=3); fi
            for depth in 200 4096 16384 32768 49 1129; do
                rows=8 extra=() nseq=1
                case $depth in 200) rows=64;; 49|1129) rows=32 extra=(interleave) nseq=2;; esac
                for ver in frozen candidate; do
                    lib=$frozen
                    if [[ $ver == candidate ]]; then lib="$PWD/build/bin"; fi
                    f="$work/$ver-$setting-$depth.bin"
                    env "${flags[@]}" LD_LIBRARY_PATH="$lib" "$work/logits" "$Q2" perf/ppl.txt "$depth" "$rows" "${extra[@]}" >"$f" 2>"$f.log"
                    test "$(stat -c %s "$f")" -eq "$((rows*nseq*200058*4))"
                    sha256sum "$f"
                done
                cmp "$work/frozen-$setting-$depth.bin" "$work/candidate-$setting-$depth.bin"
            done
        done >"$out/full-logits.txt" ;;
    ops)
        for ops in MUL_MAT,MUL_MAT_ID INKLING_ATTN,INKLING_ROUTER,SHORTCONV_RES,RES_MIX,LOOP_CTRL_DOWN,LOOP_CTRL_APPLY SITU CONCAT; do
            args=()
            if [[ $ops == CONCAT ]]; then args=(-p 'n_slots=|n_used='); fi
            build/bin/test-backend-ops test -b CUDA0 -o "$ops" "${args[@]}" >"$work/ops-$ops.log" 2>&1
            echo "$ops"
            rg 'tests passed|Backend CUDA0.*OK' "$work/ops-$ops.log" | sed -E 's/\x1B\[[0-9;]*[mK]//g'
        done >"$out/ops.txt"
        for n in "${VERIFY_MATRIX_OPS:-3777}" 162 5 44; do rg -q " $n/$n tests passed" "$out/ops.txt"; done ;;
    sync)
        for setting in default mr; do
            flags=()
            if [[ $setting == mr ]]; then flags=(IK_MR=1 IK_MMA=3); fi
            for depth in 113 1129; do
                log="$work/sync-$setting-$depth"
                env "${flags[@]}" GGML_CUDA_DISABLE_GRAPHS=1 "$HOME/opt/cuda-12.8/bin/compute-sanitizer" --tool synccheck --error-exitcode 86 \
                    --kernel-name kns=ik_dense_gate_scan --kernel-name kns=ik_dense_warp_roles --kernel-name kns=ik_ctrl_apply_pack \
                    --log-file "$log.log" "$work/logits" "$Q2" perf/ppl.txt "$depth" 2 >"$log.bin" 2>"$log-model.log"
                test "$(stat -c %s "$log.bin")" -eq "$((2*200058*4))"
                echo "$setting prefix $depth"
                rg 'ERROR SUMMARY: 0 errors' "$log.log"
            done
        done >"$out/synccheck.txt" ;;
    snapshot)
        release_env
        snapshot=${3:?snapshot directory}
        export SPEC_SERVER_BIN="$snapshot/llama-server" LD_LIBRARY_PATH="$snapshot"
        export SPEC_KINDS=edit,prose SPEC_MAXT=384 SPEC_REPS=5 SPEC_OUT="$out/snapshot-probes"
        mkdir -p "$SPEC_OUT"
        for label in plain-before n3 n3-adapt plain-after; do
            args=(--backend-sampling)
            flags=(IK_MR=1 IK_MMA=3)
            if [[ $label == n3* ]]; then
                args+=(--spec-type draft-dspark --spec-draft-model "$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf" --spec-draft-n-max 3 --spec-draft-p-min 0)
                if [[ $label == n3-adapt ]]; then flags+=(LLAMA_DSPARK_ADAPTIVE=1); fi
            fi
            env -u LLAMA_DSPARK_ADAPTIVE "${flags[@]}" perf/spec_probe.py "$label" -- "${args[@]}" | tee "$SPEC_OUT/$label.txt"
        done
        python3 perf/p6/r4_cmp.py "$SPEC_OUT" plain-before n3 n3-adapt plain-after >"$SPEC_OUT/compare.txt"
        for k in edit prose; do
            for ext in ids json; do cmp "$SPEC_OUT/plain-before.$k.$ext" "$out/384/plain-mr.$k.$ext"; done
        done ;;
    matrix)
        release_env
        group=${3:?main, prose or fixed} budget=${4:?384 or 3000}
        export SPEC_MAXT=$budget SPEC_REPS=1
        if [[ $budget == 384 ]]; then export SPEC_REPS=3; else test "$budget" = 3000; fi
        case "$group" in
            main) d=$budget; rows=(n2 n3 n4 n2-adapt n3-adapt n4-adapt);;
            fixed) d=fixed-$budget; export LLAMA_DSPARK_POLICY=0; rows=(n2 n3 n4);;
            prose)
                d=prose-3000
                if [[ $budget == 384 ]]; then d=prose; export SPEC_REPS=5; fi
                export SPEC_PROMPTS_FILE=perf/p7b/prose-prompts.json SPEC_KINDS=letter,science,scene,review,dialogue
                rows=(n2-adapt n3-adapt n4-adapt);;
            *) exit 2;;
        esac
        bash perf/p6/r4_spec.sh "$out/$d" plain-mr "${rows[@]}"
        python3 perf/p6/r4_cmp.py "$out/$d" plain-mr "${rows[@]}" >"$out/$d/compare.txt"
        for k in ${SPEC_KINDS//,/ }; do
            for ext in ids json; do cmp "$out/$d/plain-mr.$k.$ext" "perf/p7c/release/$d/plain-mr.$k.$ext"; done
        done ;;
    *) exit 2 ;;
esac
