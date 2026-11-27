#!/usr/bin/env bash
# One long-context edit, including process cleanup inside the GPU lock.
set -euo pipefail
[[ ${CUDA_VISIBLE_DEVICES:?} == 0 && ${P8_GPU:?} == 0 ]]
source perf/p8bc/env.sh
ctx=$1 depth=$2 width=$3 label=$4
export W="$PWD/build/p11/long"
mkdir -p "$W" perf/p11/b/long
out="$PWD/perf/p11/b/long/$label"
export LD_LIBRARY_PATH="${B_LIBS:-$HOME/inkling-live-bin-p9-72fd097}"
model=${B_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf}
extra=()
mode=plain
if ((width)); then
    mode=dspark
    extra=(--spec-type draft-dspark --spec-draft-model "$P8_DRAFT" --spec-draft-n-max "$width" --spec-draft-n-ubatch 512 --spec-draft-p-min 0)
fi
export LLAMA_DSPARK_ADAPTIVE=1
python3 - "$W/client.py" <<'PY'
from pathlib import Path
import sys
doc = Path('perf/PHASE10_NUMBERS.md').read_text()
code = doc.split('cat >"$W/long.py" <<\'PYTHON\'\n', 1)[1].split('\nPYTHON\n', 1)[0]
Path(sys.argv[1]).write_text(code.replace('127.0.0.1:18471', '127.0.0.1:18551'))
PY
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits -lms 200 >"$W/$label-$mode.vram" &
monitor=$!
"$LD_LIBRARY_PATH/llama-server" -m "$model" --mmproj "$HOME/models/inkling-2-mini/mmproj-Inkling-2-Mini-Q8_0.gguf" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 -b 2048 -ub 512 -c "$ctx" --backend-sampling --alias inkling-mini --host 127.0.0.1 --port 18551 "${extra[@]}" >"$W/$label-$mode.server" 2>&1 &
pid=$!
trap 'kill "$pid" "$monitor" 2>/dev/null || :; wait "$pid" || :; wait "$monitor" || :' EXIT
for ((i=0; i<300; i++)); do
    kill -0 "$pid"
    if curl -fsS --max-time 1 http://127.0.0.1:18551/health >/dev/null 2>&1; then break; fi
    sleep 0.5
done
curl -fsS http://127.0.0.1:18551/health >/dev/null
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader | awk -v p="$pid" '$1+0 == p {print; found=1} END {exit !found}' >"$out-$mode-ready.txt"
python3 "$W/client.py" "$label" "$mode" edit "$depth" | sed 's/[[:blank:]]*$//' >"$out-$mode.txt"
awk -F', ' '{if ($2+0>peak) peak=$2; if (NR==1 || $3+0<low) low=$3} END {print "samples",NR,"peak used MiB",peak,"min free MiB",low}' "$W/$label-$mode.vram" >>"$out-$mode.txt"
