"""Collect completed, chronologically ordered native operator crossovers."""
import csv
import hashlib
import io
import re
import statistics
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

screen = Path('perf/p11/b/screen')
sources = [('expert', Path('build/p11/native-expert-replays')), ('expert', screen / 'micro'),
           ('attention', screen / 'scan')]
legs = ('old1', 'new1', 'new2', 'old2')
freed = datetime.fromisoformat('2026-11-27T03:30:22-05:00').timestamp()
seen, raw = set(), {}
table = io.StringIO()
writer = csv.writer(table, delimiter='\t', lineterminator='\n')
writer.writerow(('op', 'family', 'shape', 'old1_us', 'new1_us', 'new2_us', 'old2_us',
                 'old_mean_us', 'new_mean_us', 'delta_pct', 'equal', 'old_output_sha256', 'new_output_sha256'))
counts = Counter()
for op, directory in sources:
    for p in sorted(directory.glob('*-old1-*.tsv')):
        prefix, shape = p.name.split('-old1-')
        key = (op, prefix, shape.removesuffix('.tsv'))
        if key in seen:
            continue
        files = [directory / f'{prefix}-{leg}-{shape}' for leg in legs]
        if any(not f.is_file() or not f.stat().st_size for f in files):
            continue
        stamps = [f.stat().st_mtime_ns for f in files]
        if stamps != sorted(stamps) or stamps[0] < freed * 1e9:
            continue
        records = [f.read_text().split() for f in files]
        hashes = [r[-1] for r in records]
        assert all(re.fullmatch('[a-f0-9]{64}', x) for x in hashes)
        assert all(len(x) == (13 if op == 'expert' else 12) for x in records)
        times = [statistics.median(map(float, r[-10:-1])) for r in records]
        assert min(times) > 0
        old, new = statistics.mean(times[::3]), statistics.mean(times[1:3])
        same = len(set(hashes)) == 1
        writer.writerow((*key, *(round(t, 4) for t in (*times, old, new, 100*(new/old-1))), int(same), hashes[0], hashes[1]))
        for f in files:
            raw[f'{op}/{f.name}'] = f.read_bytes()
        counts[op, same] += 1
        seen.add(key)
dest = screen / 'native-brackets.tsv'
dest.write_text(table.getvalue())
raw[dest.name] = dest.read_bytes()
archive = screen / 'native-brackets.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for name, data in sorted(raw.items()):
        meta = tarfile.TarInfo(name)
        meta.size = len(data)
        t.addfile(meta, io.BytesIO(data))
with tarfile.open(archive) as t:
    assert all(hashlib.sha256(t.extractfile(n).read()).digest() == hashlib.sha256(d).digest() for n, d in raw.items())
print('Native paired operator cells', dict(counts))
