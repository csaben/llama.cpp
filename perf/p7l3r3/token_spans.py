"""Separate the host token boundary from GPU work and within-token gaps on a serial stream."""
import sqlite3
import sys
from pathlib import Path
from statistics import mean

for path in sys.argv[1:]:
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX, k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
        """).fetchall()
    heads = [i for i,r in enumerate(rows) if r[0] == 'mul_mat_vec_q' and r[3] == 100512][-120:]
    assert len(heads) == 120
    spans = []
    for lo, hi in zip(heads, heads[1:]):
        token = rows[lo:hi]
        assert len({r[4] for r in token}) == 1
        elapsed = rows[hi][1] - rows[lo][1]
        busy = sum(end-start for _,start,end,_,_ in token)
        softcap, = (i for i,r in enumerate(token) if r[0] == 'softcap_f32')
        boundary = token[softcap+1][1] - token[softcap][2]
        spans.append((elapsed,busy,boundary,elapsed-busy-boundary))
    print(f'{Path(path).name:27} us/token total, kernel work, host boundary, other signed gaps:',
          *(f'{mean(col)/1000:.3f}' for col in zip(*spans)))
