# Inkling 2 Mini on one RTX 3090, p9

The largest tested slots to complete a 3,000-token DSpark edit were **196,608 tokens for 2-bit** and **81,920 for both 4-bit variants**. At the common 65,536-token edit depth, DSpark decoded at **272.3** (2-bit), **210.2** (IQ4_NL dense) and **178.3 tok/s** (Q6_K dense).

Measured 2026-11-22 on GPU1 (PCI 03:00.0), driver 580.173.02, existing 420 W limit and 24,576 MiB physical VRAM. GPU1 used 1,310 MiB before testing. VRAM figures include other allocations on the card. The release uses one slot, Q8_0 K/V, Flash Attention, batch 2,048 and microbatch 512. The nine snapshot files matched the p9 manifest at source `72fd097e`. The two primary targets used the release Q8_0 shortlist-32K drafter. Their IQ4_NL `output.weight` tensors have the same shape, type and SHA-256 (`b965bd17519ca4ee155bb14e16f59760f61d2fa0fa07b41cee6b14fbf1248a44`), and the token, token-type and merge tables match.

All rates are tokens/s. Plain is the mean of three 128-token synthetic decode runs. DSpark is one uncached, greedy, fixed-length 3,000-token chat completion per cell with adaptive draft cap 6. Short rates are **code / edit / prose / agent replay** at 47 / 1,025 / 36 / 2,025 prompt tokens. A dash marks an unmeasured cell. The 4K and 16K prefill cells are three-run synthetic means. Full-slot synthetic prefill is one pass. Served prefill uses the server's prompt timing on the indicated edit.

### 2-bit, experimental Q2K-Q2_0-fit (9.94 GB)

| Depth or slot (tokens) | Plain decode | DSpark decode | Prefill |
|---|---:|---:|---:|
| Empty / short | 229.2 | 278.6 / 310.4 / 307.1 / 329.5 | - |
| 4,096 | 217.9 | - | 3,531.2 |
| 16,384 | 204.6 | - | 3,212.0 |
| 32,768 | 188.4 | - | - |
| 65,536 | 159.0 | 272.3 | 2,199.2 served |
| 131,072 | 120.5 | 212.2 | 1,662.5 served |
| 196,608 slot | 91.6 at 196,480 | 189.2 at 193,600 | 1,455.6 full / 1,347.0 served |

Same-prompt plain served decode was **156.8 / 119.0 / 93.6** at the three long edit depths. 262,144 failed to allocate the drafter's compute buffers. At 196,608, GPU1 used 22,075 MiB with 2,050 MiB free at readiness. Peak sampled during the edit series was 22,243 MiB used with 1,882 MiB free. At temperature 1.0 and 1,200 tokens, code / edit ran at **231.1 / 246.8** (seed 11) and **235.4 / 260.8** (seed 22).

### 4-bit, IQ4_NL experts and dense (17.07 GB)

| Depth or slot (tokens) | Plain decode | DSpark decode | Prefill |
|---|---:|---:|---:|
| Empty / short | 192.1 | 217.2 / 222.9 / 217.8 / 246.7 | - |
| 4,096 | 183.7 | - | 3,381.5 |
| 16,384 | 174.6 | 244.3 at 16,044 | 3,092.2 |
| 32,768 | 162.6 | 225.5 at 32,045 | - |
| 65,536 | 140.8 | 210.2 | 2,137.8 served |
| 81,920 slot | 127.1 at 81,792 | 216.6 at 78,911 | 2,182.3 full / 2,001.2 served |
| 98,176 | 121.9 | OOM at 98,304 slot | - |
| 131,072 | 109.7 | OOM at startup | - |

Same-prompt plain served decode was **138.9 / 129.6** at the two long edit depths. 262,144 and 196,608 failed during target setup. At 131,072 the drafter failed to load. At 98,304 the server passed its health check (24,029 MiB used / 96 free), fell back to host feature extraction and hit CUDA graph allocation OOM on the 95,296-token edit (24,103 used / 22 free). At 81,920 the edits completed, with a sampled peak of 23,557 MiB used / 568 free. At temperature 1.0 and 1,200 tokens, code / edit ran at **192.4 / 185.5** (seed 11) and **200.5 / 195.1** (seed 22).

### 4-bit, IQ4_NL experts and Q6_K dense (17.36 GB)

| Depth or slot (tokens) | Plain decode | DSpark decode | Prefill |
|---|---:|---:|---:|
| Empty / short | 169.5 | 194.0 / 197.5 / 192.8 / 204.5 | - |
| 4,096 | 160.9 | - | 3,282.8 |
| 16,384 | 153.1 | 197.0 at 16,044 | 3,005.1 |
| 32,768 | 143.9 | 191.3 at 32,045 | - |
| 65,536 | 125.4 | 178.3 | 2,083.1 served |
| 81,920 slot | 115.1 at 81,792 | 175.1 at 78,911 | 2,140.1 full / 1,967.3 served |
| 98,176 | 109.3 | OOM at 98,304 slot | - |
| 131,072 | 98.9 | OOM at startup | - |

Same-prompt plain served decode was **123.8 / 116.3** at the two long edit depths. The Q6_K output head differs from the two primary targets. Its drafter was folded from this target with the same shortlist ranking and size. All four startup probes from 98,304 through 262,144 failed. The 81,920-token slot completed both long edits with a sampled peak of 23,849 MiB used / 276 free. At temperature 1.0 and 1,200 tokens, code / edit ran at **179.3 / 221.4** (seed 11) and **174.4 / 194.5** (seed 22).

### Correctness

- **Greedy output.** Against a fresh same-flags plain server, 2-bit matched on four short workloads and three long edits (21,000 IDs). At the live width, each 4-bit variant matched on six release cases and two long edits (24,000 IDs each). In the complete 3,000-token p9 matrix, each 4-bit variant matched all 80 drafted replies and 240,000 IDs (42 main, 20 additional prose, 18 fixed-policy). Comparisons covered the whole assistant message, including reasoning, and every emitted token ID.
- **Window exactness.** Every `mr_check` compared 182 logit rows over 64 steps with partial rollback. Zero rows differed, with maximum absolute logit difference 0. The 2-bit prefixes were 131,072 and 196,352. Both 4-bit targets passed at 200, 81,664 and 131,072 with the target-only checker.
- **Needle retrieval.** Three labeled codes appeared at 10%, 50% and 90% of the source-token stream. Plain and DSpark returned all three in order in their final answers at 131,072 and 193,600 prompt tokens (2-bit), 81,400 (IQ4_NL dense) and 78,912 (Q6_K dense). Each pair matched on its full message and IDs. Both 4-bit targets also returned all three in plain mode at 131,072, where the drafter exceeded VRAM. The Q6_K-dense plain server completed that request in a 132,096-token slot after a 134,144-token slot ran out of VRAM during prefill.
- **Reply budget.** At 196,088 prompt tokens (2-bit) and 81,400 (Q6_K dense), the 512-token reply cap left the final answer empty. Plain and DSpark identified all three labeled codes in reasoning and emitted identical full messages and IDs. The 193,600- and 78,912-token trials above reserved 3,000 tokens per reply within the same maximum slots.

<details>
<summary>Reproduce the measurements (Bash, from the repository root)</summary>

Run on a free benchmark port. The snapshot supplies the server and runtime libraries. The matching p9 build directory supplies the bench frontend and `mr_check`. Raw logs, JSON, output IDs and VRAM samples from this run are in `build/phase10/` on the benchmark host. Use a fresh output directory on replay.

```bash
set -euo pipefail
source perf/p5env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SNAP="$HOME/inkling-live-bin-p9-72fd097" W="$(mktemp -d)"
export LD_LIBRARY_PATH="$SNAP:$PWD/build/bin"
export IK_MR=1 IK_MMA=3 IK_WINDOW_COLS=1 IK_QUEUE_INPUTS=1 LLAMA_DSPARK_DEV=1 LLAMA_DSPARK_ROUND=1 LLAMA_DSPARK_GRAPHS=1 LLAMA_DSPARK_ADAPTIVE=1 LLAMA_DSPARK_POLICY=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0
export FOLD="${P10_FOLD:-$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf}"
common=(-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 -b 2048 -ub 512 --override-tensor token_embd.weight=CUDA0 --backend-sampling --host 127.0.0.1 --port 18471 --alias inkling-mini)
spec=(--spec-type draft-dspark --spec-draft-model "$FOLD" --spec-draft-override-tensor mask_embd.weight=CUDA0 --spec-draft-n-ubatch 512 --spec-draft-n-max 6 --spec-draft-p-min 0)
(cd "$SNAP" && sha256sum -c "$OLDPWD/perf/p9/snapshot.sha256")
nvidia-smi -i 1 -q -d POWER
ldd build/bin/llama-bench
PYTHONPATH=gguf-py .venv/bin/python - "$Q2" "$IQ4" "$Q6" <<'PYTHON'
import hashlib, sys, gguf
for path in sys.argv[1:]:
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.name == "output.weight")
    print(path, t.tensor_type.name, list(t.shape), hashlib.sha256(t.data).hexdigest())
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"):
        h = hashlib.sha256()
        for part in r.fields[key].parts:
            h.update(part)
        print(key, h.hexdigest())
PYTHON
bench() {
    local model=$1
    shift
    build/bin/llama-bench -m "$model" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ot token_embd.weight=CUDA0 -b 2048 -ub 512 -o json --progress "$@"
}
bench "$Q2"  -p 4096,16384 -n 0 -r 3
bench "$IQ4" -p 4096,16384 -n 0 -r 3
bench "$Q2"  -p 0 -n 128 -d 0,4096,16384,32768,65536,131072,196480 -r 3
bench "$IQ4" -p 0 -n 128 -d 0,4096,16384,32768,65536,98176 -r 3
bench "$IQ4" -p 0 -n 128 -d 81792 -r 3
bench "$Q2"  -p 196608 -n 0 -r 1 --no-warmup
bench "$IQ4" -p 81920 -n 0 -r 1 --no-warmup
bench "$IQ4" -p 0 -n 128 -d 131072 -r 3

capacity() (
set -eu
model=$1
draft=$2
spec[3]=$draft
for ctx in 262144 196608 131072 98304 65536 32768; do
    log="$W/cap-$(basename "$model")-$ctx"
    "$SNAP/llama-server" -m "$model" -c "$ctx" "${common[@]}" "${spec[@]}" >"$log" 2>&1 &
    pid=$!
    trap 'kill "$pid" 2>/dev/null || :; wait "$pid" || :' EXIT
    for i in {1..300}; do
        kill -0 "$pid" 2>/dev/null || break
        if curl -fsS --max-time 1 http://127.0.0.1:18471/health >/dev/null 2>&1; then
            echo "$ctx ready, pid=$pid"
            nvidia-smi -i 1 --query-gpu=memory.used,memory.free --format=csv
            nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader | awk -v p="$pid" '$1+0==p'
            exit 0
        fi
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then echo 'startup timeout'; exit 2; fi
    wait "$pid" || :
    trap - EXIT
    echo "$ctx startup failed"
    tail -n 5 "$log"
done
exit 1
)

capacity "$Q2" "$FOLD"
capacity "$IQ4" "$FOLD"

matrix() (
set -euo pipefail
export MODEL=$1
label=$2
mode=$3
export SPEC_OUT="$W/$label" SPEC_SERVER_BIN="$SNAP/llama-server" SPEC_SERVER_LOG="$W/$label-$mode.log" SPEC_CTX=65536 SPEC_PORT=18471 SPEC_FIXED_LENGTH=1 SPEC_TOKEN_IDS=1 SPEC_MAXT=3000 SPEC_REPS=1 SPEC_KINDS=${4:-code,edit,prose,agent}
mkdir -p "$SPEC_OUT"
opts=()
if [[ $mode == dspark ]]; then opts=("${spec[@]}"); fi
perf/spec_probe.py "$mode" -- -b 2048 -ub 512 --override-tensor token_embd.weight=CUDA0 --backend-sampling --alias inkling-mini "${opts[@]}" | tee "$SPEC_OUT/$mode.txt"
)

sampled() (
set -euo pipefail
export MODEL=$1 SPEC_OUT="$W/$2-temp" SPEC_SERVER_BIN="$SNAP/llama-server" SPEC_CTX=65536 SPEC_PORT=18471 SPEC_FIXED_LENGTH=1 SPEC_MAXT=1200 SPEC_REPS=1 SPEC_KINDS=code,edit
mkdir -p "$SPEC_OUT"
for seed in 11 22; do
    export SPEC_SERVER_LOG="$SPEC_OUT/$seed.log"
    python3 - "$seed" -- -b 2048 -ub 512 --override-tensor token_embd.weight=CUDA0 --backend-sampling --alias inkling-mini "${spec[@]}" <<'PY' | tee "$SPEC_OUT/$seed.txt"
from pathlib import Path
import sys
p = Path("perf/spec_probe.py").resolve()
src = p.read_text()
key = '"temperature": 0, "seed": 1,'
assert src.count(key) == 1
src = src.replace(key, f'"temperature": 1.0, "seed": {int(sys.argv[1])},')
exec(compile(src, str(p), "exec"), {"__file__": str(p), "__name__": "__main__"})
PY
done
)

matrix "$Q2" q2 dspark
matrix "$IQ4" iq4 dspark code,edit,prose,edit@16K,edit@32K,agent
sampled "$Q2" q2
sampled "$IQ4" iq4

cat >"$W/long.py" <<'PYTHON'
import json
import os
import sys
import urllib.request
from pathlib import Path

def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:18471" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2700) as response:
        return json.load(response)

def tokenize(s):
    return post("/tokenize", {"content": s})["tokens"]

def detok(ids):
    return post("/detokenize", {"tokens": ids})["content"]

w = Path(os.environ["W"])
label, mode, kind, depth = sys.argv[1:]
depth = int(depth)
budget = 3000 if kind == "edit" else int(os.getenv("NEEDLE_BUDGET", "512"))
stem = f"{kind}-{depth}"
prompt_file = w / (stem + ".prompt.json")
keys = ["juniper", "marigold", "cobalt"]
values = ["XQ-4718", "VB-9026", "KM-3581"]
if not prompt_file.exists():
    src = "".join(f"\n// ===== {p.name} =====\n" + p.read_text() for p in sorted(Path("src").glob("llama-*.cpp")))
    ids = tokenize(src)
    n = depth - 64
    for _ in range(12):
        assert 0 < n <= len(ids)
        if kind == "edit":
            s = detok(ids[:n]) + "\n\nCopy the first function defined in the source above exactly, then add a one-line comment above each of its statements. Output only the code."
        else:
            chunks = []
            last = 0
            for frac, key, value in zip([10, 50, 90], keys, values):
                end = n * frac // 100
                chunks.extend([detok(ids[last:end]), f"\n// Archive record: the {key} access code is {value}.\n"])
                last = end
            chunks.extend([detok(ids[last:n]), "\n\nRead the three archive records in the source. Return the juniper, marigold and cobalt access codes in that order, verbatim."])
            s = "".join(chunks)
        messages = [{"role": "user", "content": s}]
        formatted = post("/apply-template", {"messages": messages})["prompt"]
        actual = len(post("/tokenize", {"content": formatted, "add_special": True, "parse_special": True})["tokens"])
        if actual == depth:
            break
        n += depth - actual
    assert abs(depth - actual) <= 2, (depth, actual)
    prompt_file.write_text(json.dumps(messages))
body = dict(model="inkling-mini", messages=json.loads(prompt_file.read_text()), max_tokens=budget,
            temperature=0, seed=1, cache_prompt=False, ignore_eos=kind == "edit", verbose=True, return_tokens=True)
response = post("/v1/chat/completions", body)
timings = response["timings"]
msg = response["choices"][0]["message"]
ids = response["__verbose"]["tokens"]
assert abs(timings["prompt_n"] - depth) <= 2 and len(ids) == timings["predicted_n"] and ids, timings
assert kind != "edit" or len(ids) == 3000
assert any(msg.get(k) for k in ("content", "reasoning_content", "tool_calls"))
record = dict(message=msg, ids=ids, timings=timings)
if kind == "needle":
    import re
    record["recalled"] = re.findall(r"\b[A-Z]{2}-[0-9]{4}\b", msg.get("content") or "")
dest = w / f"{label}-{mode}-{stem}.json"
dest.write_text(json.dumps(record))
print(stem, {k: timings.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}, record.get("recalled", ""), flush=True)
other = w / f"{label}-{'plain' if mode == 'dspark' else 'dspark'}-{stem}.json"
if other.exists():
    ref = json.loads(other.read_text())
    print("full message equal", ref["message"] == msg, "emitted IDs equal", ref["ids"] == ids, flush=True)
    assert ref["message"] == msg and ref["ids"] == ids
if kind == "needle":
    if record["recalled"] == values:
        print("Final answer returned all three codes in order.")
    elif not msg.get("content") and len(ids) == budget and all(
        f"{key} access code is {value}" in (msg.get("reasoning_content") or "") for key, value in zip(keys, values)
    ):
        print("Reply cap reached during reasoning; all three labeled codes found there. Final-answer retrieval is censored.")
    else:
        raise ValueError(f"Incorrect retrieval: {msg.get('content')!r}")
PYTHON

long_run() (
set -euo pipefail
model=$1 label=$2 mode=$3 ctx=$4
shift 4
base="$W/$label-$mode-$1-$2"
opts=()
if [[ $mode == dspark ]]; then opts=("${spec[@]}"); fi
nvidia-smi -i 1 --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits -lms 200 >"$base.vram" &
monitor=$!
"$SNAP/llama-server" -m "$model" -c "$ctx" "${common[@]}" "${opts[@]}" >"$base.server" 2>&1 &
pid=$!
trap 'kill "$pid" "$monitor" 2>/dev/null || :; wait "$pid" || :; wait "$monitor" || :' EXIT
for i in {1..300}; do
    kill -0 "$pid" || exit 1
    if curl -fsS --max-time 1 http://127.0.0.1:18471/health >/dev/null 2>&1; then break; fi
    sleep 0.5
done
while (( $# )); do
    python3 "$W/long.py" "$label" "$mode" "$1" "$2" || exit 1
    shift 2
done
)

long_run "$Q2" q2 dspark 196608 edit 65536 edit 131072 edit 193600
long_run "$IQ4" iq4 dspark 98304 edit 95296 || rg -q 'CUDA error: out of memory' "$W/iq4-dspark-edit-95296.server"
long_run "$IQ4" iq4 dspark 81920 edit 65536 edit 78912

mr() (
set -euo pipefail
python3 -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(Path("perf/ppl.txt").read_bytes()*12)' "$W/mr-input"
MR_RS_A=0 MR_RS=6 MR_SIZES=2,3,4,5,6,7 MR_RAND=37 MR_PLAIN_PCT=0 MR_CTX="$3" build/bin/mr_check "$1" "$W/mr-input" "$4" 64 2>&1 | tee "$W/$2-mr-$4.log"
)

sweep() (
set -euo pipefail
export P8_GPU=1 P8_LIBS="$SNAP" P10_MODEL="$1" P10_DRAFT="$FOLD" SPEC_SERVER_BIN="$SNAP/llama-server"
label=$2
dir="$W/$label-matrix"
mkdir -p "$dir"
sed -e 's/for budget in 384 3000;/for budget in 3000;/' \
    -e '/^\. perf\/p8bc\/env.sh$/a export MODEL="$P10_MODEL" P8_DRAFT="$P10_DRAFT" SPEC_PORT=18471' \
    -e '/bash perf\/p8bc\/probe.sh/s/$/ -b 2048 -ub 512 --spec-draft-n-ubatch 512 --alias inkling-mini/' \
    perf/p8bc/matrix.sh >"$dir/commands.sh"
bash -c "$(cat "$dir/commands.sh")" perf/p8bc/matrix.sh "$dir"
)

matrix "$IQ4" iq4 plain code,edit,prose,edit@16K,edit@32K,agent
SPEC_KINDS=code,edit,prose,edit@16K,edit@32K,agent SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$W/iq4" plain dspark
long_run "$IQ4" iq4 plain 81920 edit 65536 edit 78912
long_run "$Q2" q2 plain 196608 edit 65536 edit 131072 edit 193600
mr "$IQ4" iq4 8192 200
mr "$IQ4" iq4 81920 81664
mr "$Q2" q2 196608 131072
mr "$Q2" q2 196608 196352
long_run "$Q2" q2 plain 196608 needle 131072
long_run "$Q2" q2 dspark 196608 needle 131072
for mode in plain dspark; do long_run "$Q2" q2 "$mode" 196608 needle 196088; done
for mode in plain dspark; do NEEDLE_BUDGET=3000 long_run "$Q2" q2 "$mode" 196608 needle 193600; done
for mode in plain dspark; do long_run "$IQ4" iq4 "$mode" 81920 needle 81400; done
NEEDLE_BUDGET=3000 long_run "$IQ4" iq4 plain 134144 needle 131072
mr "$IQ4" iq4 132096 131072
matrix "$Q2" q2 plain
SPEC_KINDS=code,edit,prose,agent SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$W/q2" plain dspark
OPENBLAS_NUM_THREADS=4 .venv/bin/python perf/p6/dspark_fold.py "$Q6" "$DRAFT" "$W/dspark-q6-short32k.gguf" perf/p6/shortlist_rank.bin 32768
capacity "$Q6" "$W/dspark-q6-short32k.gguf"
sweep "$IQ4" iq4
export P10_FOLD="$W/dspark-q6-short32k.gguf" FOLD="$W/dspark-q6-short32k.gguf"
spec[3]="$FOLD"
bench "$Q6" -p 4096,16384 -n 0 -r 3
bench "$Q6" -p 0 -n 128 -d 0,4096,16384,32768,65536,81792 -r 3
matrix "$Q6" q6 dspark code,edit,prose,edit@16K,edit@32K,agent
sampled "$Q6" q6
long_run "$Q6" q6 dspark 81920 edit 65536 edit 78912
bench "$Q6" -p 81920 -n 0 -r 1 --no-warmup
matrix "$Q6" q6 plain code,edit,prose,edit@16K,edit@32K,agent
SPEC_KINDS=code,edit,prose,edit@16K,edit@32K,agent SPEC_REQUIRE_IDS=1 python3 perf/p6/r4_cmp.py "$W/q6" plain dspark
long_run "$Q6" q6 plain 81920 edit 65536 edit 78912
mr "$Q6" q6 8192 200
mr "$Q6" q6 81920 81664
for mode in plain dspark; do NEEDLE_BUDGET=3000 long_run "$Q6" q6 "$mode" 81920 needle 78912; done
bench "$Q6" -p 0 -n 128 -d 98176,131072 -r 3
NEEDLE_BUDGET=3000 long_run "$Q6" q6 plain 134144 needle 131072 || rg -q 'CUDA error: out of memory' "$W/q6-plain-needle-131072.server"
NEEDLE_BUDGET=1000 long_run "$Q6" q6 plain 132096 needle 131072
mr "$Q6" q6 132096 131072
for mode in plain dspark; do long_run "$Q6" q6 "$mode" 81920 needle 81400; done
sweep "$Q6" q6

```

</details>
