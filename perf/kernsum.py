#!/usr/bin/env python3
# summarize an nsys gpukernsum csv per token (token count from the flash-attention launches / 48)
import csv, re, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
FA_PER_TOKEN = int(sys.argv[2]) if len(sys.argv) > 2 else 48
tot = sum(float(r["Total Time (ns)"]) for r in rows); inst = sum(int(r["Instances"]) for r in rows)
NT = sum(int(r["Instances"]) for r in rows if "flash_attn_ext" in r["Name"]) / FA_PER_TOKEN
print(f"tokens {NT:.0f}: GPU kernel time/token {tot/1e6/NT:.2f} ms, launches/token {inst/NT:.0f}")
agg = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    name = r["Name"]
    name = name[5:] if name.startswith("void ") else name
    name = re.sub(r'<.*', '', name.split('(')[0])
    agg[name][0] += int(r["Instances"]); agg[name][1] += float(r["Total Time (ns)"])
for k, v in sorted(agg.items(), key=lambda x: -x[1][1])[:int(sys.argv[3]) if len(sys.argv) > 3 else 28]:
    print(f"{v[1]/1e6/NT:7.3f} ms/tok {v[0]/NT:7.1f}/tok  {v[1]/v[0]/1e3:6.2f} us avg  {k}")
