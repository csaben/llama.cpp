"""Extract launch geometry and register use from the warmed single-token part of a trace."""
import sqlite3
import sys
from pathlib import Path

for path in sys.argv[1:]:
    with sqlite3.connect(path) as db:
        heads = db.execute("""
            SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
            WHERE s.value='mul_mat_vec_q' AND k.gridX=100512 ORDER BY k.start DESC LIMIT 120
        """).fetchall()
        assert len(heads) == 120
        rows = db.execute("""
            SELECT DISTINCT s.value, k.gridX, k.gridY, k.blockX, k.registersPerThread,
                k.staticSharedMemory, k.dynamicSharedMemory, k.localMemoryPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
            WHERE k.start BETWEEN ? AND ? AND s.value IN ('ik_complete_layer','ik_route_experts',
                'inkling_router2_kernel','ik_dense_scan_inputs','ik_dense_gate_scan','ik_projection_conv_kernel',
                'ik_q2_down_rows','mul_mat_vec_q','ik_mix_down','ik_ctrl_apply_pack','ik_controlled_router')
            ORDER BY s.value, k.gridX, k.gridY, k.registersPerThread
        """, (heads[-1][0], heads[0][0])).fetchall()
    print(Path(path).name, 'grid_x grid_y threads_x registers/thread static_shared_bytes dynamic_shared_bytes local_bytes/thread')
    for name, *fields in rows:
        print(f'{name:28}', *(f'{v:7}' for v in fields))
