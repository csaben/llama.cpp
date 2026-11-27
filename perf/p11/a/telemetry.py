"""Summarize the physical cards sampled during each complete paired benchmark."""
import csv
import re
import sys
from pathlib import Path

root = Path("build/p11a/telemetry")
print("label\tcard\tsamples\tboard_peak_MiB\tmin_temp_C\tmax_temp_C\tactive_sm_MHz_min\tactive_sm_MHz_max\tmemory_MHz_min\tmemory_MHz_max\tinitial_compute_PIDs")
devices = {"GPU-afe18a20-ce24-20de-9935-7b37fbd27581": 1, "GPU-a5900811-5e3c-f130-cec4-7b2867cd8203": 0}
labels = set()
for group in sys.argv[1:]:
    labels.update(path.stem for path in root.glob(f"{group}-*.csv"))
for label in sorted(labels):
    if not re.match(r"quiet-(?:batch-)?(?:olda|newa|newb|oldb)-", label):
        continue
    residents = (root / f"{label}.residents").read_text()
    assert "1358328" not in residents, label
    resident_rows = re.findall(r"(GPU-[a-z0-9-]+),\s*(\d+),", residents)
    with (root / f"{label}.csv").open() as handle:
        records = list(csv.reader(handle, skipinitialspace=True))
    for card in sorted({int(r[1]) for r in records}):
        pids = ",".join(pid for uuid, pid in resident_rows if devices[uuid] == card) or "none"
        rows = [list(map(int, r[2:])) for r in records if int(r[1]) == card]
        active = [r for r in rows if r[4] > 0]
        clocks = [r[2] for r in active]
        print(label, card, len(rows), max(r[0] for r in rows), min(r[1] for r in rows), max(r[1] for r in rows),
              min(clocks, default="unsampled"), max(clocks, default="unsampled"), min(r[3] for r in rows), max(r[3] for r in rows), pids, sep="\t")
