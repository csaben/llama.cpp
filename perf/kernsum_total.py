#!/usr/bin/env python3
# summarize an nsys gpukernsum csv by kernel (totals, for prefill profiles): perf/kernsum_total.py <csv> [rows]
import csv, re, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"total GPU kernel time {tot/1e6:.1f} ms, {sum(int(r['Instances']) for r in rows)} launches")
agg = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    n = r["Name"]; n = n[5:] if n.startswith("void ") else n; n = re.sub(r'<.*', '', n.split('(')[0])
    agg[n][0] += int(r["Instances"]); agg[n][1] += float(r["Total Time (ns)"])
for k, v in sorted(agg.items(), key=lambda x: -x[1][1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 20]:
    print(f"{v[1]/1e6:9.1f} ms {100*v[1]/tot:5.1f}%  {v[0]:6d} x {v[1]/v[0]/1e3:8.1f} us  {k}")
