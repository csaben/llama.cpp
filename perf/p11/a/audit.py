"""Audit the Phase 9 suites against a fresh, same-model p9 control."""
import csv
import hashlib
import re
import sys
from pathlib import Path

root, reference = map(Path, sys.argv[1:])
keys = ("suite", "budget", "prompt", "setting")

def cells(directory):
    with (directory / "cells.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(rows) != len(result) or len(result) != 194:
        raise ValueError(f"{directory} has {len(rows)} cells, {len(result)} unique")
    return result

rows, old = cells(root), cells(reference)
assert rows.keys() == old.keys()
replies = ids = 0
for (suite, budget, prompt, setting), cell in rows.items():
    here = root / suite / budget
    p9 = reference / suite / budget
    reps = (5 if suite == "prose" else 3) if budget == "384" else 1
    line, = [s for s in (here / f"{setting}.txt").read_text().splitlines() if s.split()[1] == prompt]
    if reps > 1:
        assert len(re.fullmatch(r".* runs ([0-9/]+)", line)[1].split("/")) == reps, line
    for ext, key in (("ids", "ids_sha256"), ("json", "message_sha256")):
        data = (here / f"{setting}.{prompt}.{ext}").read_bytes()
        assert data == (here / f"plain.{prompt}.{ext}").read_bytes(), (suite, budget, prompt, setting, ext)
        assert data == (p9 / f"plain.{prompt}.{ext}").read_bytes(), (suite, budget, prompt, setting, ext)
        assert hashlib.sha256(data).hexdigest() == cell[key] == old[suite, budget, prompt, setting][key]
        if ext == "ids":
            assert len(data) == 4*int(budget)
    if setting != "plain":
        replies += reps
        ids += reps*int(budget)
assert (replies, ids) == (360, 347520)
print(f"{len(rows)} cells, {replies} drafted replies, {ids:,} IDs and complete messages match same-flags plain and p9. Every short repeat agreed.")
