"""Audit all four near-full requests and summarize their timing and residency."""
import json
import re
import statistics
import sys
from pathlib import Path

name = sys.argv[1]
base = Path(__file__).parent / "capacity"
physical = {"GPU-afe18a20-ce24-20de-9935-7b37fbd27581": 1, "GPU-a5900811-5e3c-f130-cec4-7b2867cd8203": 0}
rows = []
for leg in ("olda", "newa", "newb", "oldb"):
    label = f"{name}-{leg}-128"
    for ext in ("ids", "json"):
        assert (base / f"{label}.{ext}").read_bytes() == (base / f"p9-128-plain.{ext}").read_bytes()
    timings = json.loads((base / f"{label}.timings").read_text())
    assert timings["prompt_n"] == 128064 and timings["predicted_n"] == 3000
    mem = (base / f"{label}.memory").read_text()
    per_process, board = [0, 0], [None, None]
    for uuid, value in re.findall(r"^process peak MiB (GPU-[a-z0-9-]+) (\d+)$", mem, re.M):
        per_process[physical[uuid]] = int(value)
    for uuid, value in re.findall(r"^(GPU-[a-z0-9-]+), \d+, (\d+) MiB$", mem, re.M):
        card = physical[uuid]
        per_process[card] = max(int(value), per_process[card])
    for card, value in re.findall(r"^physical ([01]) peak used MiB (\d+)", mem, re.M):
        board[int(card)] = int(value)
    assert per_process[1] and board[1]
    if leg.startswith("new"):
        assert per_process[0] and board[0]
    else:
        assert per_process[0] == 0 and board[0] is None
    rows.append([leg, timings["prompt_per_second"], timings["predicted_per_second"], *per_process, *board])
print("leg\tprompt_tok_s\tdecode_tok_s\tprocess_GPU0_MiB\tprocess_GPU1_MiB\tboard_GPU0_MiB\tboard_GPU1_MiB")
for data in rows:
    print(data[0], f"{data[1]:.2f}", f"{data[2]:.2f}", *[v if v is not None else "unsampled" for v in data[3:]], sep="\t")
for role, group in (("p9 mean", (rows[0], rows[3])), ("candidate mean", rows[1:3])):
    print(role, *(f"{statistics.mean(r[i] for r in group):.2f}" for i in (1,2)), sep="\t")
