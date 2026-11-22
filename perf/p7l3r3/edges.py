"""Rank adjacent kernels on one CUDA stream over complete warmed decode steps."""
import collections
import sqlite3
import statistics
import sys

with sqlite3.connect(sys.argv[1]) as db:
    rows = db.execute("""
        SELECT s.value, k.gridX, k.gridY, k.start, k.end, k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
    """).fetchall()

heads = [i for i, row in enumerate(rows) if row[:3] == ("mul_mat_vec_q", 100512, 1)]
assert len(heads) >= 120
pairs = collections.defaultdict(list)
ops = collections.defaultdict(list)
for lo, hi in zip(heads[-120:-1], heads[-119:]):
    assert len({r[-1] for r in rows[lo:hi+1]}) == 1
    for prev, cur in zip(rows[lo:hi], rows[lo+1:hi+1]):
        def label(row):
            return f"{row[0]}[{row[1]},{row[2]}]" if row[0] == "mul_mat_vec_q" else row[0]
        gap = cur[3] - prev[4]
        cost = cur[4] - cur[3]
        ops[label(cur)].append(cost)
        pairs[label(prev), label(cur)].append((gap, cost))

print("119 warmed head-to-head intervals; output head included; signed gaps in us")
print("kernel  calls/token  busy_us/call  busy_us/token")
for op, vals in sorted(ops.items(), key=lambda item: -sum(item[1])):
    print(f"{op:36} {len(vals)/119:5.1f} {statistics.mean(vals)/1000:8.3f} {sum(vals)/119000:9.3f}")
print("producer -> consumer  calls/token  mean_gap_us  mean_consumer_us  total_gap_us/token")
for edge, spans in sorted(pairs.items(), key=lambda item: -sum(a for a, b in item[1])):
    a, b = zip(*spans)
    print(f"{edge[0]:27} -> {edge[1]:27} {len(a)/119:5.1f} {statistics.mean(a)/1000:8.3f} {statistics.mean(b)/1000:8.3f} {sum(a)/119000:9.3f}")
