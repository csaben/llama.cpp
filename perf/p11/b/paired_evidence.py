"""Archive completed ABBA cells after checking the raw rates, plain replies and synthetic controls."""
import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
work = Path('build/p11')
pair = work / 'narrow-paired'
evidence = Path('perf/p11/b/narrow')
legs = ('old1', 'new1', 'new2', 'old2')
files = []
rows = []


def append(workload, prompt, rates, digest):
    old = statistics.mean((rates[0], rates[3]))
    new = statistics.mean(rates[1:3])
    rows.append([workload, prompt, *(round(x, 3) for x in (*rates, old, new, 100*(new/old-1))), digest])


for cell in ('tg0', 'tg4096', 'tg16384', 'tg32768', 'tg65536', 'pp4096', 'pp16384', 'pp65536'):
    captures = [pair / f'{cell}-{leg}.json' for leg in legs]
    proof = evidence / 'bench-logits' / f'{cell}.txt'
    if not all(p.is_file() and p.stat().st_size for p in [*captures, proof]):
        continue
    data = [json.loads(p.read_bytes())[0] for p in captures]
    for key in ('model_filename', 'n_depth', 'n_prompt', 'n_gen', 'n_batch', 'n_ubatch', 'type_k', 'type_v'):
        assert len({d[key] for d in data}) == 1, (cell, key)
    for d in data:
        depth, prompt, gen = (int(cell[2:]), 0, 128) if cell.startswith('tg') else (0, int(cell[2:]), 0)
        assert (d['n_depth'], d['n_prompt'], d['n_gen']) == (depth, prompt, gen)
        assert d['avg_ts'] > 0 and len(d['samples_ts']) == 3
    digest = re.search(r'^Full logit SHA-256 ([0-9a-f]{64})$', proof.read_text(), re.M)[1]
    append(cell, '-', [d['avg_ts'] for d in data], digest)
    files.extend((p, p.name) for p in captures)
    files.append((proof, f'bench-logits/{cell}.txt'))

for cell in ('short384', 'short3000', 'deep384', 'deep3000'):
    directory = pair / cell
    budget = cell.removeprefix('short').removeprefix('deep')
    prompts = ('code', 'edit', 'prose', 'agent') if cell.startswith('short') else ('edit@16K', 'edit@32K')
    if not all((directory / f'{leg}.equal').is_file() and (directory / f'{leg}.equal').stat().st_size for leg in legs[1:]):
        continue
    tables = []
    for leg in legs:
        table = directory / f'{leg}.txt'
        records = [s.split() for s in table.read_text().splitlines()]
        assert len(records) == len(prompts) and {r[1] for r in records} == set(prompts)
        for r in records:
            assert r[0] == leg and r[4] == budget and 'accepted/proposed' in r
            if budget == '384':
                assert r[-2] == 'runs' and len(r[-1].split('/')) == 3
        tables.append({r[1]: float(r[6]) for r in records})
        files.append((table, f'{cell}/{table.name}'))
    for prompt in prompts:
        for ext in ('json', 'ids'):
            controls = [work / f'matrix-{version}/main/{budget}/plain.{prompt}.{ext}' for version in ('p9', 'narrow')]
            captures = [directory / f'{leg}.{prompt}.{ext}' for leg in legs]
            expected = controls[0].read_bytes()
            assert all(p.read_bytes() == expected for p in [controls[1], *captures]), (cell, prompt, ext)
            if ext == 'ids':
                assert len(expected) == int(budget)*4
                digest = hashlib.sha256(expected).hexdigest()
            else:
                msg = json.loads(expected)
                assert any(msg.get(key) for key in ('content', 'reasoning_content', 'tool_calls'))
            for p in captures:
                files.append((p, f'{cell}/{p.name}'))
            for version, p in zip(('p9', 'narrow'), controls):
                files.append((p, f'{cell}/{version}-plain.{prompt}.{ext}'))
        append(cell, prompt, [t[prompt] for t in tables], digest)

assert args.partial or len(rows) == 20, f'{len(rows)} of 20 paired output comparisons complete'
if rows:
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t', lineterminator='\n')
    writer.writerow(('workload', 'prompt', *legs, 'old_mean', 'new_mean', 'delta_pct', 'output_sha256'))
    writer.writerows(rows)
    table = evidence / 'paired.tsv'
    table.write_text(output.getvalue())
    with tarfile.open(evidence / 'paired-captures.tar.gz', 'w:gz') as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print(f'{len(rows)} output-verified paired comparisons archived.')
