"""Archive fully paired native runtime screens with frozen plain outputs."""
import csv
import hashlib
import io
import json
import statistics
import sys
import tarfile
from datetime import datetime
from pathlib import Path

root = Path('build/p11')
out = Path('perf/p11/b/screen')
name, cell = sys.argv[1:]
dirs = {'packed': 'packed-bracket', 'cost': 'paired', 'cost-prototype': 'policy-bracket-native'}
runs = root / dirs[name] / cell
budget = int(cell.removeprefix('short').removeprefix('deep'))
assert budget in (384, 3000)
prompts = ('edit@16K', 'edit@32K') if cell.startswith('deep') else ('code', 'edit', 'prose', 'agent')
legs = ('old1', 'new1', 'new2', 'old2')
rates, raw = [], {}
times = []
for leg in legs:
    summary = runs / (leg+'.txt')
    times.append(summary.stat().st_mtime)
    lines = [s.split() for s in summary.read_text().splitlines()]
    assert {r[1] for r in lines} == set(prompts) and len(lines) == len(prompts)
    rates.append({r[1]: float(r[6]) for r in lines})
    raw[summary.name] = summary.read_bytes()
    for r in lines:
        assert r[0] == leg and int(r[4]) == budget and 'accepted/proposed' in r
        if budget == 384:
            assert r[-2] == 'runs' and len(r[-1].split('/')) == 3
assert times == sorted(times) and times[0] > datetime.fromisoformat('2026-11-27T03:30:22-05:00').timestamp()
text = io.StringIO()
writer = csv.writer(text, delimiter='\t', lineterminator='\n')
writer.writerow(('prompt', *legs, 'old_mean', 'new_mean', 'delta_pct', 'ids_sha256', 'message_sha256'))
for prompt in prompts:
    hashes = []
    for ext in ('ids', 'json'):
        ref = root / f'matrix-p9/main/{budget}/plain.{prompt}.{ext}'
        answer = ref.read_bytes()
        if ext == 'ids':
            assert len(answer) == 4*budget
        else:
            assert any(json.loads(answer).get(k) for k in ('content', 'reasoning_content', 'tool_calls'))
        raw[ref.name] = answer
        for leg in legs:
            f = runs / f'{leg}.{prompt}.{ext}'
            assert f.read_bytes() == answer
            raw[f.name] = answer
        hashes.append(hashlib.sha256(answer).hexdigest())
    values = [r[prompt] for r in rates]
    old, new = statistics.mean(values[::3]), statistics.mean(values[1:3])
    writer.writerow([prompt, *(round(x, 3) for x in (*values, old, new, 100*(new/old-1))), *hashes])
stem = f'native-{name}-{cell}'
(out / (stem+'.tsv')).write_text(text.getvalue())
raw['summary.tsv'] = text.getvalue().encode()
with tarfile.open(out / (stem+'.tar.gz'), 'w:gz') as t:
    for key, data in sorted(raw.items()):
        info = tarfile.TarInfo(key)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
with tarfile.open(out / (stem+'.tar.gz')) as t:
    assert all(t.extractfile(k).read() == v for k, v in raw.items())
print(text.getvalue(), end='')
