#!/usr/bin/env python3
# usage: ksplit.py <nsys sqlite> <name prefix> -> launches, mean us and registers per (demangled name, grid, block)
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); pre = sys.argv[2]
h = collections.Counter(); t = collections.Counter()
for name, us, gx, gy, bx, by, reg in c.execute("select d.value,(k.end-k.start)/1000.0,k.gridX,k.gridY,k.blockX,k.blockY,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id join StringIds d on k.demangledName=d.id where s.value like ?", (pre + '%',)):
    key = (name[:90], gx, gy, bx, by, reg); h[key] += 1; t[key] += us
for k, v in sorted(t.items(), key=lambda x: -x[1])[:12]:
    print(f"{h[k]:7d} x {v/h[k]:7.2f} us  {k}")
