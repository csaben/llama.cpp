"""Rank the kernels in a real tg128 decode step using logical tensor traffic."""
import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

E = 2304
F = 4 * E
Q = 36 * (E // 32)
HEAD = 201024
IQ4 = 18 * (E // 32)


def footprint(name, params, grid, pos, full):
    """Logical input and output bytes. Count a shared input once per launch."""
    gx, gy, _ = grid
    if name.startswith(('ik_dense_gate_scan', 'ik_dense_warp_roles')):
        taps = params[0]
        # The current cell is written once. The scan uses only visible historical cells.
        cells = pos if taps else min(pos, 1023)
        proj = (5528 + 64 * taps) * 4
        clip_conv = (3072 + 384 + 2 * 1024 * 4) * 4
        table = min(cells + 1, 1024) * 16 * 4
        history = 2 * 1024 * 3 * 4 * 2
        kv = (cells + 1) * 2 * 8 * 4 * 34
        mask_pos = (cells + 1) * 6
        return 3072 * IQ4 + Q + proj + clip_conv + table + history + kv + mask_pos + 96 * 36 + 12 + 4 * taps
    if name == 'fill_kernel':
        # The initial state row has 19,968 F32 entries.
        return 3 * (2*1024 + 2*E) * 4
    if name == 'ik_dense_scan_inputs':
        return gx * 4 * (IQ4 + 4) + Q * (3 if full else 1)
    if name == 'ik_projection_conv_kernel':
        # Omit optional residual input for a conservative bound.
        return E * (3072 // 32 * 18) + 96 * 36 + F * (4 + 3 + 3 + 1) + 4
    if name == 'ik_mix_down':
        cols = 3 if full and gy == 3 else 1
        return F * (params[0] + 2 * cols + gy) + gy * 64 * (2 * E * 2 + 4)
    if name == 'loop_ctrl_down_f32':
        batches = grid[2]
        cols = 3 if full and batches == 3 else 1
        return F * (cols + batches) + batches * 64 * (2 * E * 2 + 4)
    if name in ('ik_ctrl_apply_pack', 'ik_ctrl_apply8'):
        cols = 3 if full and gy == 3 else 1
        weights = gy * 2 * E * 64 * 2
        io = F * (cols + gy) + gy * 64 * 4
        return weights + io + (F * cols + Q * gy if name == 'ik_ctrl_apply_pack' else F * gy)
    if name == 'inkling_router2_kernel':
        ns = params[0]
        return F * (128 + ns + (ns > 1) + 1 + 1) + Q + 128 * 4 + 16 * 4
    if name == 'res_mix_f32_vec4':
        _, q8, ns = params[:3]
        queries = 0 if ns == 1 else gy
        norm = gy if q8 or ns == 5 else 0
        return F * (queries + ns + norm) + (Q if q8 else F) * gy
    if name == 'inkling_moe_sum_conv_kernel':
        return (8 + 4 + 3 + 3 + 1 + 1) * F + 8 * 4 + 4
    if name == 'situ_q8_1_kernel':
        return 8 * 1152 * 8 + 8 * 36 * 36
    if name == 'rms_norm_f32':
        return 3 * F
    if name == 'quantize_q8_1':
        return F + (2560 // 32) * 36
    if name == 'k_get_rows_float':
        return 2 * F + 4
    if name == 'softcap_f32':
        return 2 * HEAD * 4
    if name == 'ik_q2_down_rows':
        return 5971968 + 8 * (36 * 36 + F + 4)
    if name == 'mul_mat_vec_q':
        if gy == 1:
            return HEAD * (IQ4 + 4) + Q
        if params[0] == 10:
            return 13934592 + Q + 8 * (F + 4)
        if gx == 1152:
            return 23887872 + Q + 8 * (F + 4)
        return 11943936 + 8 * (36 * 36 + F + 4)
    raise ValueError(name)


def measurements(path, depth, steps):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        rows = db.execute("""
            SELECT s.value, d.value, k.end-k.start, k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
            JOIN StringIds d ON d.id=k.demangledName ORDER BY k.start
        """).fetchall()
    ends = [i for i, r in enumerate(rows) if r[0] == 'softcap_f32'][-129:]
    assert len(ends) == 129
    groups = defaultdict(list)
    for step in steps:
        # The last 128 softcaps are the 128 timed generation tokens.
        end = ends[-128 + step]
        begin = ends[-128 + step - 1] + 1
        token = rows[begin:end + 1]
        dense = [i for i, r in enumerate(token) if r[0] == 'ik_dense_scan_inputs']
        assert len(dense) == 48
        for i, (name, demangled, ns, *shape) in enumerate(token):
            p = tuple(map(int, re.findall(r'\((?:int|bool|ggml_type)\)(-?\d+)', demangled)))
            grid, block = tuple(shape[:3]), tuple(shape[3:])
            nxt = next((j for j in dense if j >= i), dense[0])
            full = token[nxt][3] == 1398
            b = footprint(name, p, grid, depth + step, full)
            key = name, p, grid, block
            groups[key].append((ns / 1000, b))
    return groups


def report(path, depth, steps):
    groups = measurements(path, depth, steps)
    rankings = []
    families = defaultdict(lambda: [0., 0., 0.])
    for (name, params, grid, block), samples in groups.items():
        calls = len(samples) / len(steps)
        us, size = map(mean, zip(*samples))
        floor = 2.5 + size / 889000
        excess = (us - floor) * calls
        rankings.append((excess, name, params, grid, block, calls, us, size, floor))
        for j, value in enumerate((calls, us*calls, excess)):
            families[name][j] += value
    pure = {'mul_mat_vec_q', 'ik_dense_scan_inputs', 'ik_q2_down_rows'}
    print(f'## Depth {depth}, generation indices {steps[0]} through {steps[-1]}\n')
    print('| Kernel family | Calls/token | us/token | Excess us/token |')
    print('|---|---:|---:|---:|')
    for name, (calls, total, excess) in sorted(families.items(), key=lambda item: -item[1][2]):
        if name not in pure:
            print(f'| {name} | {calls:g} | {total:.2f} | {excess:.2f} |')
    print('\n| Kernel (template arguments) | Grid | Block | Calls/token | us/call | us/token | Logical B/call | Floor us/call | Excess us/token |')
    print('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for excess, name, params, grid, block, calls, us, size, floor in sorted(rankings, reverse=True):
        cfg = lambda dims: 'x'.join(map(str, dims))
        print(f'| {name}{params or ""} | {cfg(grid)} | {cfg(block)} | {calls:g} | {us:.3f} | {us*calls:.2f} | {size:,.0f} | {floor:.3f} | {excess:.2f} |')
    print(f'\n{sum(row[0] for row in families.values()):g} launches/token.')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('trace', type=Path)
    cli.add_argument('depth', type=int)
    cli.add_argument('--mean', action='store_true')
    cli.add_argument('--step', type=int, default=64, choices=range(128), metavar='0..127')
    args = cli.parse_args()
    report(args.trace, args.depth, range(8, 128) if args.mean else [args.step])
