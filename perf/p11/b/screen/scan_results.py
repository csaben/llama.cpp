"""Summarize the selected attention windows and preserve the timed groups."""
import csv
import statistics
import sys
from pathlib import Path

directory = Path(sys.argv[1])
rows = {}
for path in sorted(directory.glob('*.tsv')):
    name, depth, width = path.stem.rsplit('-', 2)
    data = path.read_text().split()
    assert data[:2] == [depth, width] and len(data) == 12, path
    rows[name, int(depth), int(width)] = (list(map(float, data[2:-1])), data[-1])

writer = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
if len(sys.argv) > 2 and sys.argv[2] == 'paired':
    writer.writerow(('depth','width','old1_us','new1_us','new2_us','old2_us','old_us','new_us','delta_pct','matching_captures'))
    for name, depth, width in sorted(rows):
        if name != 'packed-old1':
            continue
        tests = [rows['packed-'+leg, depth, width] for leg in ('old1','new1','new2','old2')]
        times = [statistics.median(t[0]) for t in tests]
        old = statistics.mean((times[0],times[3]))
        new = statistics.mean(times[1:3])
        writer.writerow((depth,width,*times,round(old,3),round(new,3),round(100*(new/old-1),2),len({t[1] for t in tests}) == 1))
else:
    writer.writerow(('variant','depth','width','median_us','output_sha256','same_as_p9',*[f'group_{i}_us' for i in range(1,10)]))
    for (name, depth, width), (times, digest) in sorted(rows.items()):
        writer.writerow((name,depth,width,statistics.median(times),digest,digest==rows['base',depth,width][1],*times))
