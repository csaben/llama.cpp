#!/usr/bin/env bash
# p9 rollback, boundary, width and PPL checks on the selected runtime and target.
set -euo pipefail
d=${1:?output directory}
work=$(mktemp -d build/p11/checks.XXXXXX)
trap 'rm -rf "$work"' EXIT
mkdir -p "$d"
gpu=(perf/p11/b/gpu.sh perf/p11/b/run.sh)
export MR_RS_A=0 MR_RS=4
gzip -dc perf/p6/r4_repro/n3.trace.gz >"$work/n3.trace"
mr() {
    local name=$1
    shift
    if ! rg -q ' 0 rows differ from single-token decode, max \|logit diff\| 0$' "$d/$name.log" 2>/dev/null; then
        "${gpu[@]}" mr "$@" >"$d/$name.log" 2>&1
    fi
    rg 'rows differ from single-token decode, max \|logit diff\| 0$' "$d/$name.log" | tee -a "$d/rows.txt"
}
: >"$d/rows.txt"
MR_RS=3 MR_CTX=16384 MR_TRACE="$PWD/$work/n3.trace" mr trace perf/ppl.txt
for depth in 1533 3069 4500; do
    MR_RAND=2 MR_SIZES=5 MR_PLAIN_PCT=0 mr "boundary-$depth" perf/ppl.txt "$depth" 24
done
IK_ATTN_FIXED_MR=0 MR_RAND=2 MR_SIZES=5 mr legacy-3069 perf/ppl.txt 3069 24
for depth in 2000 2046 2558 3069 4094 4500; do
    MR_RAND=37 MR_SIZES=2,3,4,5 MR_PLAIN_PCT=15 MR_PLAIN_RUN=8 mr "rollback-$depth" perf/ppl.txt "$depth" 64
done
python3 - "$d/rows.txt" <<'PY'
import re, sys
rows = re.findall(r'(\d+) (?:output )?rows (?:up to position|in \d+ steps).*?: (\d+) rows differ', open(sys.argv[1]).read())
assert len(rows) == 11 and sum(int(n) for n, _ in rows) == 2714 and all(b == '0' for _, b in rows), rows
PY
cat perf/ppl.txt perf/ppl.txt >"$work/depth-input"
for pack in 0 1; do
    for depth in 200 4096 16384 32768; do
        MR_RS=6 MR_SIZES=2,3,4,5,6,7 MR_RAND=37 MR_PLAIN_PCT=0 MR_CTX=34816 IK_WINDOW_COLS=$pack mr "wide-$pack-$depth" "$work/depth-input" "$depth" 64
    done
done
for width in 2 3 4 5 6 7; do
    MR_RS=6 MR_SIZES=$width MR_ROLLBACK=0 mr "cost-$width" perf/ppl.txt 200 32
    rg 'single token:|window of ' "$d/cost-$width.log" >"$d/cost-$width.txt"
done
for mode in prefill decode; do
    opts=(-c 2048 -b 2048 -ub 512 --chunks 8)
    if [[ $mode == decode ]]; then opts=(-c 1024 -b 1024 -ub 1 --chunks 2); fi
    "${gpu[@]}" ppl "${opts[@]}" >"$d/$mode.log" 2>&1
    rg 'Final estimate:' "$d/$mode.log" >"$d/$mode.txt"
done
