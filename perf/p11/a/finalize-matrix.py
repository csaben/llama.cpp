"""Verify the run configuration and select the six post-residency control measurements."""
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

repo = Path.cwd()
root = Path("perf/p11/a/release-matrix")
reference = Path("perf/p11/a/p9-matrix")
repeated = Path("perf/p11/a/replay/release")
work = Path("build/p11a")
cutoff = datetime.datetime.fromisoformat("2026-11-27T03:30:00-05:00").timestamp()
runtime = (work / "repro-direct/bin/llama-server").resolve()
settings = {
    "main": ("n2", "n3", "n4", "n2-adapt", "n3-adapt", "n4-adapt", "n6-adapt"),
    "prose": ("n2-adapt", "n3-adapt", "n4-adapt", "n6-adapt"),
    "fixed": ("n2", "n3", "n4"),
}
rows = ["suite\tbudget\tsetting\tmode\tplacement\tctx\ttarget_ub\tdraft_ub\tnative\tdirect\tserver"]
controls = []
for suite, labels in settings.items():
    for budget in ("384", "3000"):
        suffix = Path(suite) / budget
        for label in ("plain", *labels):
            logs = work / ("replay/release" if label == "plain" else "matrix/release-matrix") / suffix
            config = logs / f"{label}.server.config"
            assert (logs / f"{label}.done").is_file() and config.stat().st_mtime >= cutoff, config
            fields = config.read_text().strip().split("\t")
            assert fields == ["plain" if label == "plain" else "draft", "head", "131072", "512", "512", "1", "1", str(runtime)], config
            rows.append("\t".join((suite, budget, label, *fields)))
        originals = list((root / suffix).glob("plain.*.ids"))
        assert len(originals) == (5 if suite == "prose" else 6), suffix
        for item in originals:
            for ext in ("ids", "json"):
                src = item.with_suffix("." + ext)
                assert src.read_bytes() == (repeated / suffix / src.name).read_bytes() == (reference / suffix / src.name).read_bytes(), src
        controls.append(suffix)

for suffix in controls:
    backup = work / "matrix-early-control-times" / suffix / "plain.txt"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / suffix / "plain.txt", backup)
    shutil.copy2(repeated / suffix / "plain.txt", root / suffix / "plain.txt")
cells = subprocess.check_output([sys.executable, "perf/p8bc/summarize.py", root])
(root / "cells.tsv").write_bytes(cells)
audit = subprocess.check_output([sys.executable, "perf/p11/a/audit.py", root, reference, repeated])
assert len(rows) == 35
(root / "placement.tsv").write_text("\n".join(rows) + "\n")
(root / "audit.txt").write_bytes(audit)
sys.stdout.buffer.write(audit)
