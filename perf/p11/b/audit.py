"""Compare the release matrix with fresh p9 controls on the same GGUF."""
import csv
import re
import sys
from pathlib import Path

root, frozen = map(Path, sys.argv[1:3])


def read_cells(path):
    with (path / 'cells.tsv').open() as source:
        return list(csv.DictReader(source, delimiter='\t'))


refs = {(r['suite'], r['budget'], r['prompt']): r for r in read_cells(frozen) if r['setting'] == 'plain'}
rows = read_cells(root)
replies = ids = 0
for row in rows:
    suite, budget, prompt, setting = (row[k] for k in ('suite', 'budget', 'prompt', 'setting'))
    old = refs[suite, budget, prompt]
    for ext, key in [('ids', 'ids_sha256'), ('json', 'message_sha256')]:
        assert row[key] == old[key], (suite, budget, prompt, setting, key)
        expected = (frozen / suite / budget / f'plain.{prompt}.{ext}').read_bytes()
        data = (root / suite / budget / f'{setting}.{prompt}.{ext}').read_bytes()
        assert data == expected, (suite, budget, prompt, setting, ext)
    reps = (5 if suite == 'prose' else 3) if budget == '384' else 1
    lines = (root / suite / budget / f'{setting}.txt').read_text().splitlines()
    line, = [s for s in lines if s.split()[1] == prompt]
    if reps > 1:
        assert len(re.fullmatch(r'.* runs ([0-9/]+)', line)[1].split('/')) == reps, line
    if setting != 'plain':
        replies += reps
        ids += reps * int(budget)
assert (len(rows), replies, ids) == (194, 360, 347520)
print(f'{len(rows)} cells, {replies} drafted replies, {ids:,} IDs. Every message and ID matched same-flags plain and frozen p9.')
