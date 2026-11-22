"""Write one auditable TSV row per matrix cell from the Phase 8 probe captures."""
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("suite\tbudget\tsetting\tprompt\ttok/s\taccepted\tproposed\tmessage_sha256\tids_sha256")
settings = {
    "main": ["plain", "n2", "n3", "n4", "n2-adapt", "n3-adapt", "n4-adapt", "n6-adapt"],
    "prose": ["plain", "n2-adapt", "n3-adapt", "n4-adapt", "n6-adapt"],
    "fixed": ["plain", "n2", "n3", "n4"],
}
for suite, labels in settings.items():
    kinds = {"letter", "science", "scene", "review", "dialogue"} if suite == "prose" else {"code", "edit", "prose", "edit@16K", "edit@32K", "agent"}
    for budget in (384, 3000):
        directory = root / suite / str(budget)
        for setting in labels:
            table = directory / f"{setting}.txt"
            seen = set()
            for line in table.read_text().splitlines():
                label, kind, _, _, n, _, rate, *_ = line.split()
                if label != setting or kind in seen or int(n) != budget:
                    raise ValueError(line)
                seen.add(kind)
                hashes = []
                for ext in ("json", "ids"):
                    data = (directory / f"{label}.{kind}.{ext}").read_bytes()
                    if data != (directory / f"plain.{kind}.{ext}").read_bytes():
                        raise ValueError(f"{suite} {budget} {label} {kind} {ext} differs")
                    if ext == "ids" and len(data) != 4*budget:
                        raise ValueError(f"{label}.{kind}.ids length {len(data)}")
                    if ext == "json" and not any(json.loads(data).get(k) for k in ("content", "reasoning_content", "tool_calls")):
                        raise ValueError(f"{label}.{kind}.json empty")
                    hashes.append(hashlib.sha256(data).hexdigest())
                accepted = re.search(r"accepted/proposed (\d+)/(\d+)", line)
                counts = accepted.groups() if accepted else ("", "")
                print(suite, budget, label, kind, rate, *counts, *hashes, sep="\t")
            if seen != kinds:
                raise ValueError(f"{table}: expected {sorted(kinds)}, found {sorted(seen)}")
