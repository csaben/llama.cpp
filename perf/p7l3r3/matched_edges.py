"""Use visit order to compare the same FFN tail and controller sites on both paths."""
import sqlite3
import statistics
import sys
from pathlib import Path

def load(path):
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
        """).fetchall()
    heads = [i for i,r in enumerate(rows) if r[0] == 'mul_mat_vec_q' and r[3] == 100512][-120:]
    assert len(heads) == 120
    return [rows[a:b] for a,b in zip(heads, heads[1:])]

base = load(sys.argv[1])
for path in sys.argv[2:]:
    tokens = load(path)
    print(f'{Path(path).name} kernels/token {statistics.mean(map(len,tokens)):.0f} (control {statistics.mean(map(len,base)):.0f})')
    for fused, first, second in [('ik_tail_mix', 'inkling_moe_sum_conv_kernel','res_mix_f32_vec4'),
                                 ('ik_mix_down', 'res_mix_f32_vec4','loop_ctrl_down_f32')]:
        reference, candidate = [], []
        for before, after in zip(base,tokens):
            if first == 'res_mix_f32_vec4':
                sites = [i for i,r in enumerate(after) if r[0] in (fused, second)]
                controls = [i-1 for i,r in enumerate(before) if r[0] == second]
            else:
                sites = [i for i,r in enumerate(after) if r[0] in (fused, first)]
                controls = [i for i,r in enumerate(before) if r[0] == first]
            assert len(controls) == len(sites) == 48
            for i,j in zip(sites,controls):
                if after[i][0] == fused:
                    assert [before[j][0],before[j+1][0]] == [first,second]
                    reference.append(before[j+1][2]-before[j][1])
                    candidate.append(after[i][2]-after[i][1])
        if candidate:
            print(f'{fused}: {len(candidate)/119:.0f} sites/token, graph {statistics.mean(reference)/1000:.3f}, fused {statistics.mean(candidate)/1000:.3f} us/site')
