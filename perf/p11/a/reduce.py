"""Read only complete ABBA pairs and check the served replies against the p9 control."""
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).parent
name = sys.argv[1]
work = root / "paired" / name
bench = root / "paired" / sys.argv[2] if len(sys.argv) > 2 else work
legs = ["olda", "newa", "newb", "oldb"]

def emit(key, values):
    before = statistics.mean((values[0], values[3]))
    after = statistics.mean(values[1:3])
    print(key, *(f"{r:.3f}" for r in values), f"{before:.3f}", f"{after:.3f}", f"{100*(after/before-1):+.2f}", sep="\t")

print("test", *legs, "p9_mean", "candidate_mean", "change_pct", sep="\t")
for mode, depths in (("tg", (0, 4096, 16384, 32768, 65536)), ("pp", (4096, 16384, 65536))):
    for depth in depths:
        key = f"{mode}{depth}"
        vals = []
        for leg in legs:
            if bench != work:
                batch = json.loads((bench / f"{bench.name}-{leg}-{mode}.json").read_text())
                field = "n_depth" if mode == "tg" else "n_prompt"
                entry, = [row for row in batch if row[field] == depth]
            else:
                entry, = json.loads((work / f"{name}-{leg}-{key}.json").read_text())
            assert entry["n_prompt"] == (depth if mode == "pp" else 0)
            assert entry["n_gen"] == (128 if mode == "tg" else 0)
            assert entry["n_depth"] == (depth if mode == "tg" else 0)
            assert len(entry["samples_ts"]) == 3 and min(entry["samples_ts"]) > 0
            vals.append(entry["avg_ts"])
        emit(key, vals)
for budget in (384, 3000):
    key = f"edit{budget}"
    vals = []
    for leg in legs:
        label = f"{name}-{leg}-{key}"
        line, = (work / f"{label}.txt").read_text().splitlines()
        fields = line.split()
        assert fields[:2] == [label, "edit"] and int(fields[4]) == budget
        assert len(fields[-1].split("/")) == 3
        for ext in ("ids", "json"):
            data = (work / f"{label}.edit.{ext}").read_bytes()
            assert data == (root / f"p9-matrix/main/{budget}/plain.edit.{ext}").read_bytes()
            if ext == "ids":
                assert len(data) == 4*budget
        vals.append(float(fields[6]))
    emit(key, vals)
