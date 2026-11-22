"""Measure the controller critical intervals and the duration of real branch overlap."""
import collections
import sqlite3
import statistics
import sys

for path in sys.argv[1:]:
    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT s.value, k.start, k.end, k.gridX, k.gridY, k.gridZ, k.streamId, d.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
            JOIN StringIds d ON k.demangledName=d.id ORDER BY k.start
        """).fetchall()
    heads = [i for i,r in enumerate(rows) if r[0] == 'mul_mat_vec_q' and r[3] == 100512][-120:]
    assert len(heads) == 120
    times = collections.defaultdict(list)
    for lo, hi in zip(heads, heads[1:]):
        r = rows[lo:hi]
        side = [v for v in r if 'ik_ctrl_half<(bool)1>' in v[-1]]
        for v in side:
            overlap = sum(max(0, min(v[2], t[2]) - max(v[1], t[1])) for t in r if t[6] != v[6])
            times['message overlap'].append(overlap)
        main = [v for v in r if v not in side]
        for i,v in enumerate(main[:-1]):
            n = main[i+1]
            if v[0] == 'res_mix_f32_vec4' and n[0] in ('ik_ctrl_half','loop_ctrl_down_f32'):
                key = 'mix and down'
                times[key].append(n[2]-v[1])
            if v[0] == 'ik_mix_down':
                times['mix and down'].append(v[2]-v[1])
            if v[0] == 'loop_ctrl_down_f32' and n[0] == 'ik_ctrl_apply8':
                times['down and apply'].append(n[2]-v[1])
            if v[0] == 'ik_ctrl_pair':
                times['down and apply'].append(v[2]-v[1])
            if v[0] == 'ik_ctrl_apply8' and n[0] == 'rms_norm_mul_inplace_f32':
                times['apply and norm'].append(n[2]-v[1])
            if v[0] == 'ik_ctrl_apply_pack':
                times['apply and norm'].append(v[2]-v[1])
    for key, vals in times.items():
        print(f"{path.split('/')[-1]:24} {key:17} {len(vals)/119:4.1f}/token  {statistics.mean(vals)/1000:6.3f} us/call")
