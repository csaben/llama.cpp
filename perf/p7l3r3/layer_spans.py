import sys
from pathlib import Path
import sqlite3
from collections import Counter, defaultdict
from statistics import mean
def tokens(path):
    with sqlite3.connect(path) as db:
        rows=db.execute('select s.value,k.start,k.end,k.gridX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
    heads=[i for i,r in enumerate(rows) if r[0]=='mul_mat_vec_q' and r[3]==100512][-120:]
    assert len(heads)==120
    for a,b in zip(heads,heads[1:]):
        yield rows[a:b]
prefix=['ik_dense_scan_inputs','ik_dense_gate_scan','ik_projection_conv_kernel','inkling_router2_kernel','mul_mat_vec_q','situ_q8_1_kernel','ik_q2_down_rows','inkling_moe_sum_conv_kernel']
for path in sys.argv[1:]:
    name = Path(path).stem
    groups=defaultdict(list)
    for rows in tokens(path):
        for i,r in enumerate(rows):
            if r[0]==prefix[0] and [j[0] for j in rows[i:i+8]]==prefix:
                groups['complete layer'].append(rows[i+7][2]-r[1])
            elif r[0]=='ik_complete_layer': groups['complete layer'].append(r[2]-r[1])
            elif r[0]=='ik_message_dots':
                groups['message dot duration'].append(r[2]-r[1])
                overlap=Counter()
                for k in rows:
                    if k[4]!=r[4] and k[1]<r[2] and k[2]>r[1]: overlap[k[0]]+=min(r[2],k[2])-max(k[1],r[1])
                groups['total message overlap'].append(sum(overlap.values()))
                for key,value in overlap.items(): groups['overlap '+key].append(value)
    for group,vs in groups.items(): print(f'{name:10} {group:35} {len(vs)/119:5.1f} calls/token {mean(vs)/1000:7.3f} us/call')
