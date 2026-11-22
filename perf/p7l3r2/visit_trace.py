"""Time each complete dense visit between two successive vocabulary heads."""
import argparse
import collections
import sqlite3
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("trace")
args = parser.parse_args()
with sqlite3.connect(args.trace) as db:
    rows = db.execute("""
        SELECT s.value, k.gridX, k.gridY, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
    """).fetchall()

heads = [i for i, r in enumerate(rows) if r[:3] == ("mul_mat_vec_q", 100512, 1)]
if len(heads) < 120:
    raise ValueError(f"only {len(heads)} head launches")
times = collections.defaultdict(list)
for lo, hi in zip(heads[-120:-1], heads[-119:]):
    visits = rows[lo + 1:hi]
    count = 0
    for i, row in enumerate(visits):
        name = row[0]
        if name == "ik_dense_kernel":
            if visits[i + 1][0] == "ik_projection_conv_kernel":
                times["head-prefix"].append(row[4] - row[3])
                times["prefix + output"].append(visits[i + 1][4] - row[3])
            else:
                times["full visit"].append(row[4] - row[3])
        elif name == "ik_projection_conv_kernel" and visits[i - 1][0] == "inkling_attn2_kernel":
            q, attn = visits[i - 2:i]
            if q[0] not in {"mul_mat_vec_q", "ik_segmented_projection"}:
                raise ValueError(q)
            times["graph prefix"].append(attn[4] - q[3])
            times["graph visit"].append(row[4] - q[3])
            times["graph qkv"].append(q[4] - q[3])
            times["graph attention"].append(attn[4] - attn[3])
            times["graph output"].append(row[4] - row[3])
        else:
            continue
        count += 1
    if count != 48:
        raise ValueError(f"expected 48 dense visits, found {count}")
print(f"head-to-head {(rows[heads[-1]][4] - rows[heads[-120]][4])/119e6:.5f} ms")
for name, duration in times.items():
    print(f"{name:18} {len(duration)/119:5.1f} visits/token   {statistics.mean(duration)/1000:7.3f} us/visit")
