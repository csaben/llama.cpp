"""Check every merged matrix cell against its same-flags and frozen controls."""
import csv
import hashlib
import re
from pathlib import Path

root = Path("perf/p8m/matrix")
with (root / "cells.tsv").open() as f:
    cells = list(csv.DictReader(f, delimiter="\t"))
with Path("perf/p8bc/matrix/cells.tsv").open() as f:
    frozen = {(c["suite"], c["budget"], c["prompt"]): c for c in csv.DictReader(f, delimiter="\t") if c["setting"] == "plain"}
replies = ids = 0
for c in cells:
    suite, budget, prompt, setting = (c[k] for k in ("suite", "budget", "prompt", "setting"))
    d = root / suite / budget
    ref = frozen[suite, budget, prompt]
    old = {"fixed": "fixed-" + budget, "prose": "prose" if budget == "384" else "prose-3000", "main": budget}[suite]
    for ext, key in (("ids", "ids_sha256"), ("json", "message_sha256")):
        assert c[key] == ref[key], (suite, budget, prompt, setting, key)
        if prompt != "agent":
            data = (Path("perf/p8/release") / old / f"plain-mr.{prompt}.{ext}").read_bytes()
            assert hashlib.sha256(data).hexdigest() == c[key], (old, prompt, key)
    reps = (5 if suite == "prose" else 3) if budget == "384" else 1
    line, = [s for s in (d / f"{setting}.txt").read_text().splitlines() if s.split()[1] == prompt]
    if reps > 1:
        assert len(re.fullmatch(r".* runs ([0-9/]+)", line)[1].split("/")) == reps, line
    if setting != "plain":
        replies += reps
        ids += reps * int(budget)
assert (len(cells), replies, ids) == (194, 360, 347520)
print(f"{len(cells)} cells, {replies} drafted replies and {ids:,} IDs matched. All short repeats agreed.")
