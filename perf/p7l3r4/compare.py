"""Compare the last 120 complete timed steps from two matched Nsight traces."""
import argparse
from collections import defaultdict
from pathlib import Path

from latency import measurements


def totals(path, depth):
    steps = range(8, 128)
    totals = defaultdict(lambda: [0., 0., 0.])
    for (name, params, _, _), samples in measurements(path, depth, steps).items():
        if name in ('ik_dense_gate_scan', 'ik_dense_warp_roles'):
            name += ' (full)' if params[0] else ' (sliding)'
        row = totals[name]
        row[0] += len(samples)/len(steps)
        row[1] += sum(us for us, _ in samples)/len(steps)
        row[2] += sum(2.5 + size/889000 for _, size in samples)/len(steps)
    return totals


def report(control, candidate, depth):
    before, after = totals(control, depth), totals(candidate, depth)
    print(f'## Depth {depth}\n')
    print('| Kernel | Calls off/on | us/call off/on | us/token off | us/token on | Saved us/token | Excess off | Excess on |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for name in sorted(before.keys() | after.keys(), key=lambda n: before[n][1] - before[n][2], reverse=True):
        c0, t0, floor0 = before[name]
        c1, t1, floor1 = after[name]
        us0 = f'{t0/c0:.3f}' if c0 else '-'
        us1 = f'{t1/c1:.3f}' if c1 else '-'
        print(f'| {name} | {c0:g}/{c1:g} | {us0}/{us1} | {t0:.2f} | {t1:.2f} | {t0-t1:+.2f} | {t0-floor0:.2f} | {t1-floor1:.2f} |')
    t0, t1 = [sum(t for _, t, _ in data.values()) for data in (before, after)]
    print(f'\nAll kernel intervals total {t0:.2f} us/token off and {t1:.2f} on.')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('control', type=Path)
    cli.add_argument('candidate', type=Path)
    cli.add_argument('depth', type=int)
    args = cli.parse_args()
    report(args.control, args.candidate, args.depth)
