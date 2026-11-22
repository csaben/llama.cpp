"""Compare the same router-to-up and router-to-state spans, with or without fusion."""
import collections
import sqlite3
import statistics
import sys

for path in sys.argv[1:]:
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX, k.gridY
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id ORDER BY k.start
        """).fetchall()
    heads = [i for i, row in enumerate(rows) if row[0] == "mul_mat_vec_q" and row[3] == 100512]
    assert len(heads) >= 120
    spans = collections.defaultdict(list)
    for lo, hi in zip(heads[-120:-1], heads[-119:]):
        for i in range(lo, hi):
            name, start, end, *_ = rows[i]
            if name == "inkling_router2_kernel" and rows[i+1][0] == "mul_mat_vec_q" and rows[i+3][0] == "ik_q2_down_rows":
                spans["router + gate/up"].append(rows[i+1][2] - start)
                spans["five FFN stages"].append(rows[i+4][2] - start)
            elif name == "ik_route_experts":
                if rows[i+1][0] == "situ_q8_1_kernel":
                    spans["router + gate/up"].append(end - start)
                    spans["five FFN stages"].append(rows[i+3][2] - start)
                else:
                    spans["five FFN stages"].append(end - start)
    for name, v in spans.items():
        assert len(v) == 42*119
        print(f"{path.split('/')[-1]:23} {name:21} 42 visits/token  {statistics.mean(v)/1000:.3f} us/visit")
