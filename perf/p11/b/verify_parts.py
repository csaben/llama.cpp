"""Attribute the GPU time of each served p9 target graph to its main kernel groups."""
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from statistics import median

con = sqlite3.connect(f'file:{sys.argv[1]}?mode=ro', uri=True)
graph = defaultdict(Counter)
width = {}
for corr, name, grid_y, ns in con.execute("""
    SELECT k.correlationId, name.value, k.gridY, k.end - k.start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds name ON k.demangledName = name.id
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME api ON api.correlationId = k.correlationId
    JOIN StringIds call ON api.nameId = call.id
    WHERE call.value LIKE 'cudaGraphLaunch%'"""):
    if name.startswith('void ik_q2k_moe<'):
        assert grid_y % 8 == 0 and width.get(corr, grid_y//8) == grid_y//8
        width[corr] = grid_y//8
        part = 'Q2_K gate'
    elif name.startswith('void ik_mma_moe<(ggml_type)42,'):
        part = 'Q2_0 down'
    elif name.startswith('void inkling_attn_window_cols<'):
        part = 'window column scan'
    elif name.startswith('void inkling_attn2_kernel<'):
        part = 'attention main'
    else:
        part = 'other'
    graph[corr][part] += ns
parts = ['Q2_K gate', 'Q2_0 down', 'window column scan', 'attention main', 'other']
write = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n').writerow
write(['rows','launches','total GPU ms','median GPU ms'] + [p+' %' for p in parts])
for n in sorted(set(width.values())):
    graphs = [graph[k] for k in width if width[k] == n]
    grouped = sum(graphs, Counter())
    total = sum(grouped.values())
    write([n, len(graphs), f'{total/1e6:.4f}', f'{median(sum(g.values()) for g in graphs)/1e6:.4f}'] +
          [f'{100*grouped[p]/total:.2f}' for p in parts])
