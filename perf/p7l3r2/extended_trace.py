"""Compare the same direct dense-to-router visits or the Q2 expert output intervals."""
import collections
import sqlite3
import statistics
import sys

with sqlite3.connect(sys.argv[1]) as db:
    kernels = db.execute("""
        SELECT s.value,k.gridX,k.gridY,k.start,k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        ORDER BY k.start
    """).fetchall()
heads = [i for i, row in enumerate(kernels) if row[:3] == ("mul_mat_vec_q", 100512, 1)]
if len(heads) < 120:
    raise ValueError("expected at least 120 vocabulary head launches")
times = collections.defaultdict(list)
for lo, hi in zip(heads[-120:-1], heads[-119:]):
    for i in range(lo + 1, hi):
        name, _, _, begin, end = kernels[i]
        next_name, _, _, _, next_end = kernels[i + 1]
        if name == "ik_dense_late_gate" and next_name == "inkling_router2_kernel":
            times["dense + direct router"].append(next_end - begin)
        if name == "ik_resident_visit_router":
            times["dense + direct router"].append(end - begin)
        if name in {"ik_q2_down_rows", "ik_down_packed_rows"} and next_name == "inkling_moe_sum_conv_kernel":
            times["expert down + sum/conv"].append(next_end - begin)
        if name == "ik_expert_tail":
            times["expert down + sum/conv"].append(end - begin)
for key, durations in times.items():
    expected = 24 if "router" in key else 44
    assert len(durations) == 119*expected
    print(f"{key}  {expected} visits/token  {statistics.mean(durations)/1000:.3f} us/visit")
