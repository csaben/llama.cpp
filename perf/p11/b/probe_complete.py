"""Check that a finished fixed-length probe contains every requested repetition and output."""
import json
import re
import sys
from pathlib import Path

directory, label, kinds, reps, budget = sys.argv[1:]
directory = Path(directory)
summary = directory / f"{label}.txt"
if not summary.is_file():
    sys.exit(1)
lines = [s.split() for s in summary.read_text().splitlines()]
kinds = kinds.split(",")
if sorted(row[1] for row in lines if len(row) > 1) != sorted(kinds):
    sys.exit(1)
for row in lines:
    kind = row[1]
    message = directory / f"{label}.{kind}.json"
    ids = directory / f"{label}.{kind}.ids"
    if row[0] != label or row[4] != budget or not ids.is_file() or ids.stat().st_size != 4 * int(budget):
        sys.exit(1)
    if not message.is_file() or not any(v for k, v in json.loads(message.read_text()).items() if k != "role"):
        sys.exit(1)
    if int(reps) > 1 and (row[-2] != "runs" or not re.fullmatch(r"\d+(?:/\d+){" + str(int(reps)-1) + "}", row[-1])):
        sys.exit(1)
