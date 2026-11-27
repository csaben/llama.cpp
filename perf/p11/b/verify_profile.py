"""Summarize decode after the final large expert GEMM in the served edit trace."""
import csv
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
con.execute('CREATE TEMP VIEW kernels AS SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName')
cutoff, = con.execute("SELECT MAX(end) FROM kernels WHERE name LIKE 'void mul_mat_q<(ggml_type)10,%'").fetchone()
assert cutoff is not None
writer = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
writer.writerow(['GPU ms', 'calls', 'mean us', 'kernel / grid Y'])
for name, count, ns in con.execute('''
    SELECT CASE WHEN name LIKE '%ik_q2k_moe<%' OR name LIKE '%ik_mma_moe<%' THEN name || ' gridY=' || gridY ELSE name END AS label,
           COUNT(*), SUM(end-start) AS ns FROM kernels WHERE start >= ? GROUP BY label ORDER BY ns DESC''', (cutoff,)):
    writer.writerow([f'{ns / 1e6:.3f}', count, f'{ns / count / 1e3:.3f}', name])
