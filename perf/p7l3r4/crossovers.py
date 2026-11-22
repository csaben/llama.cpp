"""Summarize complete four-trial llama-bench crossovers."""
import re
import sys
from pathlib import Path

print('| Comparison | A1 | B1 | B2 | A2 | A mean | B mean | Change |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for path in map(Path, sys.argv[1:]):
    runs = [float(x) for x in re.findall(r'tg128[^\n]*?\|\s*([\d.]+)\s*±', path.read_text())]
    if len(runs) != 4:
        raise ValueError(f'{path} has {len(runs)} trials')
    a = (runs[0] + runs[3])/2
    b = (runs[1] + runs[2])/2
    values = ' | '.join(f'{r:.2f}' for r in [*runs, a, b])
    print(f'| [{path.stem}]({path.name}) | {values} | {(b/a-1)*100:+.2f}% |')
