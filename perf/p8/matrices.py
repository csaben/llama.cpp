#!/usr/bin/env python3
"""Audit the six Phase 8 release matrices and print their throughput tables."""
import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--frozen", type=Path, default=Path("perf/p7c/release"))
args = parser.parse_args()

main = ("code", "edit", "prose", "edit@16K", "edit@32K")
prose = ("letter", "science", "scene", "review", "dialogue")
widths = ("n2", "n3", "n4")
adaptive = tuple(w + "-adapt" for w in widths)
rate_line = re.compile(r"(\S+)\s+(\S+)\s+prompt\s+\d+\s+(\d+) tok\s+([\d.]+) tok/s")
drafted = token_ids = 0
print("# DSpark release matrices")
for name, budget, reps, kinds, labels in (
    ("384", 384, 3, main, widths + adaptive),
    ("3000", 3000, 1, main, widths + adaptive),
    ("prose", 384, 5, prose, adaptive),
    ("prose-3000", 3000, 1, prose, adaptive),
    ("fixed-384", 384, 3, main, widths),
    ("fixed-3000", 3000, 1, main, widths),
):
    folder = args.directory / name
    print(f"\n## {name} ({reps} request{'s' if reps != 1 else ''} per cell)\n")
    print("| Decode tok/s | " + " | ".join(kinds) + " |")
    print("|---|" + "---:|" * len(kinds))
    for label in ("plain-mr", *labels):
        rates = {}
        for line in (folder / (label + ".txt")).read_text().splitlines():
            row, kind, n, rate = rate_line.match(line).groups()
            assert row == label and int(n) == budget and kind not in rates, line
            assert reps == 1 or len(line.split(" runs ")[-1].strip().split("/")) == reps, line
            rates[kind] = rate
        assert set(rates) == set(kinds), rates
        for kind in kinds:
            ref = folder / f"plain-mr.{kind}"
            for ext in ("ids", "json"):
                sample = (folder / f"{label}.{kind}.{ext}").read_bytes()
                assert sample == Path(f"{ref}.{ext}").read_bytes(), (name, label, kind, ext)
                assert sample == (args.frozen / name / f"plain-mr.{kind}.{ext}").read_bytes(), (name, kind, ext)
                if ext == "ids":
                    assert len(sample) == 4 * budget
                else:
                    assert any(isinstance(v, str) and v for k, v in json.loads(sample).items() if k != "role")
            if label != "plain-mr":
                drafted += reps
                token_ids += reps * budget
        print("| " + label + " | " + " | ".join(rates[k] for k in kinds) + " |")
assert (drafted, token_ids) == (270, 260640)
print(f"\nAll {drafted} drafted replies and {token_ids:,} token IDs matched same-flags plain and frozen Phase 7c.")
