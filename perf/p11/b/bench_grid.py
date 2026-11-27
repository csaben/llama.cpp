"""Validate the grid's original random inputs, full-vocabulary outputs, timings and cell boundaries."""
import csv
import hashlib
import io
import json
import statistics
import sys
import tarfile
from pathlib import Path

import numpy as np

work = Path('build/p11/bench-grid')
dest = Path('perf/p11/b/narrow/bench-grid')
dest.mkdir(parents=True, exist_ok=True)
kind = sys.argv[1]
assert kind in ('tg', 'pp')
legs = ('old1', 'new1', 'new2', 'old2')
runtimes = ('old', 'new')
timed = [work / f'{kind}-timed-{leg}.json' for leg in legs]
captures = [work / f'{kind}-output-{leg}.bin' for leg in runtimes]
for file in timed + captures:
    assert file.with_suffix('.done').is_file(), file
metrics = [json.loads(f.read_bytes()) for f in timed]
for label in runtimes:
    metrics.append(json.loads((work / f'{kind}-output-{label}.json').read_bytes()))
columns = ['cell', *legs, 'old_mean', 'new_mean', 'delta_pct', 'rows', 'input_ids', 'input_sha256', 'logit_sha256']
output = io.StringIO()
writer = csv.writer(output, delimiter='\t', lineterminator='\n')
writer.writerow(columns)
with captures[0].open('rb') as old, captures[1].open('rb') as new:
    for group in zip(*metrics, strict=True):
        template = group[0]
        prompt, gen, depth = (template[k] for k in ('n_prompt', 'n_gen', 'n_depth'))
        for key in ('model_filename', 'n_prompt', 'n_gen', 'n_depth', 'n_batch', 'n_ubatch', 'type_k', 'type_v'):
            assert all(r[key] == template[key] for r in group), key
        d = depth if kind == 'tg' else prompt
        sizes = [1, *([2048] * (depth // 2048)), *([1] * (3*gen))] if kind == 'tg' else [2048] * (4*prompt//2048)
        token_hash, logit_hash = hashlib.sha256(), hashlib.sha256()
        for size in sizes:
            width = old.read(4)
            assert int.from_bytes(width, 'little') == size and new.read(4) == width
            tokens = old.read(4*size)
            assert len(tokens) == 4*size and new.read(4*size) == tokens
            data = old.read(4*200058)
            assert len(data) == 4*200058 and new.read(len(data)) == data
            assert np.isfinite(np.frombuffer(data, dtype='<f4')).all()
            token_hash.update(width + tokens)
            logit_hash.update(data)
        rates = [r['avg_ts'] for r in group[:4]]
        before, after = statistics.mean(rates[::3]), statistics.mean(rates[1:3])
        writer.writerow([f'{kind}{d}', *(round(x, 4) for x in (*rates, before, after, 100*(after/before-1))),
                         len(sizes), sum(sizes), token_hash.hexdigest(), logit_hash.hexdigest()])
    assert old.read(1) == new.read(1) == b''
table = dest / f'{kind}.tsv'
table.write_text(output.getvalue())
paths = [table, *work.glob(f'{kind}-*.json'), *work.glob(f'{kind}-*.done')]
archive = dest / f'{kind}-measurements.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for p in paths:
        t.add(p, arcname=p.name)
with tarfile.open(archive) as t:
    assert all(t.extractfile(p.name).read() == p.read_bytes() for p in paths)
print(output.getvalue(), end='')
