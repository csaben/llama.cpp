"""Summarize the two 8K IQ4_NL comparisons after checking their uncached replies."""
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
plain = {ext: (root / f"plain.edit.{ext}").read_bytes() for ext in ("ids", "json")}
assert len(plain["ids"]) == 384*4
rates = {"tg128": [], "edit384": []}
legs = ("olda", "newa", "newb", "oldb")
dates = []
for leg in legs:
    entry, = json.loads((root / f"iq4-bench-{leg}.json").read_text())
    assert entry["n_gen"] == 128 and entry["n_prompt"] == entry["n_depth"] == 0 and len(entry["samples_ts"]) == 5
    assert "IQ4_NL-dense-iq4nl" in entry["model_filename"], entry["model_filename"]
    dates.append(entry["test_time"])
    rates["tg128"].append(entry["avg_ts"])
    label = f"iq4-edit-{leg}"
    for ext, expected in plain.items():
        assert (root / f"{label}.edit.{ext}").read_bytes() == expected
    line = (root / f"{label}.txt").read_text().strip()
    fields = line.split()
    assert fields[:2] == [label, "edit"] and fields[3:6] == ["1025", "384", "tok"]
    assert re.search(r"runs \d+/\d+/\d+$", line)
    rates["edit384"].append(float(fields[6]))
assert dates == sorted(dates) and len(set(dates)) == 4
print("metric", *legs, "p9_mean", "candidate_mean", "pct", sep="\t")
for name, values in rates.items():
    old = statistics.mean((values[0], values[3]))
    new = statistics.mean(values[1:3])
    print(name, *(f"{value:.3f}" for value in (*values, old, new, 100*(new/old-1))), sep="\t")
