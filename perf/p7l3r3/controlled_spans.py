"""Match the 24 single-state FFN routers with their controller apply producers."""
import sqlite3
import sys
from pathlib import Path
from statistics import mean

for path in sys.argv[1:]:
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
        """).fetchall()
    heads = [i for i, r in enumerate(rows) if r[0] == 'mul_mat_vec_q' and r[3] == 100512][-120:]
    assert len(heads) == 120
    spans = []
    custom = 0
    for lo, hi in zip(heads, heads[1:]):
        for i in range(lo + 1, hi):
            op, start, end, _ = rows[i]
            prev = rows[i-1]
            if op == 'ik_controlled_router' or (op == 'inkling_router2_kernel' and prev[0] == 'ik_ctrl_apply8'):
                custom += op == 'ik_controlled_router'
                spans.append(end - (prev[1] if prev[0] == 'ik_ctrl_apply8' else start))
    assert len(spans) == 24*119, (len(spans), path)
    print(f'{Path(path).name:32} {len(spans)/119:.1f} controller/router pairs per token, {mean(spans)/1000:.3f} us/pair, {custom/119:.0f} specialized launches/token')
