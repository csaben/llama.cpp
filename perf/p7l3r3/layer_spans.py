"""Compare complete attention-to-FFN spans and cross-stream message work."""
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def tokens(path):
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX, k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
        """).fetchall()
    heads = [i for i, row in enumerate(rows) if row[0] == 'mul_mat_vec_q' and row[3] == 100512][-120:]
    assert len(heads) == 120
    for start, end in zip(heads, heads[1:]):
        yield rows[start:end]


chain = ['ik_dense_scan_inputs', 'ik_dense_gate_scan', 'ik_projection_conv_kernel', 'inkling_router2_kernel',
         'mul_mat_vec_q', 'situ_q8_1_kernel', 'ik_q2_down_rows', 'inkling_moe_sum_conv_kernel']
for path in sys.argv[1:]:
    groups = defaultdict(list)
    for rows in tokens(path):
        for i, (name, start, end, _, stream) in enumerate(rows):
            if name == chain[0] and [r[0] for r in rows[i:i+len(chain)]] == chain:
                groups['complete layer'].append(rows[i+len(chain)-1][2] - start)
            elif name == 'ik_complete_layer':
                groups['complete layer'].append(end - start)
            elif name == 'ik_message_dots':
                groups['message dot duration'].append(end - start)
                overlap = Counter()
                for op, other_start, other_end, _, other_stream in rows:
                    if other_stream != stream and other_start < end and other_end > start:
                        overlap[op] += min(end, other_end) - max(start, other_start)
                groups['total message overlap'].append(sum(overlap.values()))
                for op, value in overlap.items():
                    groups['overlap ' + op].append(value)
    for group, samples in groups.items():
        print(f'{Path(path).stem:10} {group:35} {len(samples)/119:5.1f} calls/token {mean(samples)/1000:7.3f} us/call')
