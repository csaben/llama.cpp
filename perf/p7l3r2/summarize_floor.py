"""Summarize matched replays and verify that each route pair used the same experts."""
import hashlib
import re
import statistics
import sys
from pathlib import Path

print("step dense_packed grid stock_ms resident_ms volatile_read_ms matched193_ms stock_repeat_ms rows mismatches max_abs route_sha256")
unique_routes = set()
for step in (0, 12, 48):
    routes = set()
    for packed in (0, 1):
        log = (Path(sys.argv[1]) / f"audit-{step}-{packed}.log").read_text()
        ids = re.findall(r"^floor op .* ids=(.+)$", log, re.M)
        assert len(ids) == 96 and ids[::2] == ids[1::2]
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        routes.add(digest)
        times = []
        for kind in ("stock-graph", "persistent", "read", "same-dot-193", "stock-repeat"):
            samples = re.findall(rf"floor time {kind} \d+ ([\d.]+) ms", log)
            assert len(samples) == 5
            times.append(f"{statistics.median(map(float, samples)):.6f}")
        grid = re.search(r"threads=\d+ grid=(\d+)", log)[1]
        check = re.search(r"check rows=(\d+) mismatches=(\d+) max_abs=([^ ]+)", log)
        assert check[2] == "0"
        print(step, packed, grid, *times, *check.groups(), digest)
    assert len(routes) == 1
    unique_routes.update(routes)
assert len(unique_routes) == 3
