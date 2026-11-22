"""Compare kernel families in the final 120 complete generation steps of node traces."""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

for path in map(Path, sys.argv[1:]):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
        """).fetchall()
    ends = [i for i, r in enumerate(rows) if r[0] == 'softcap_f32'][-121:]
    assert len(ends) == 121
    times = defaultdict(list)
    for a, b in zip(ends, ends[1:]):
        for name, start, end in rows[a + 1:b + 1]:
            times[name].append((end - start) / 1000)
    print(f'{path.stem}  launches/token {sum(map(len, times.values())) / 120:g}')
    for name, samples in sorted(times.items()):
        print(f'{name:32} {len(samples)/120:5g} {mean(samples):8.3f} {sum(samples)/120:9.3f}')
