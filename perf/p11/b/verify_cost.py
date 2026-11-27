"""Measure warmed target graph spans by verified width in a served Nsight trace."""
import sqlite3
import statistics
import sys
from collections import defaultdict

c = sqlite3.connect(sys.argv[1])
rows = c.execute("""
    SELECT k.correlationId,
           MAX(CASE WHEN name.value LIKE 'void ik_q2k_moe<%' THEN k.gridY / 8 END) AS width,
           COUNT(*), (MAX(k.end) - MIN(k.start)) / 1e6, SUM(k.end-k.start) / 1e6
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds name ON k.demangledName = name.id
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME api ON api.correlationId = k.correlationId
    JOIN StringIds call ON api.nameId = call.id
    WHERE call.value LIKE 'cudaGraphLaunch%'
    GROUP BY k.correlationId HAVING width BETWEEN 1 AND 8
    ORDER BY MIN(k.start)""")
groups = defaultdict(list)
for _, width, kernels, span, total in rows:
    groups[width].append((kernels, span, total))
print('target_rows\tgraph_launches\tnodes_min\tnodes_max\tspan_median_ms\tkernel_sum_median_ms')
for width, records in sorted(groups.items()):
    kernels, span, total = zip(*records)
    print(width, len(kernels), min(kernels), max(kernels), f'{statistics.median(span):.4f}', f'{statistics.median(total):.4f}', sep='\t')
