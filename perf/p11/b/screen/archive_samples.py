"""Preserve raw operator measurements with their original completion times."""
import argparse
import csv
import datetime
import gzip
import hashlib
import io
import re
import tarfile
from pathlib import Path
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('period', choices=('overlap', 'native'))
args = parser.parse_args()
screen = Path('perf/p11/b/screen')
work = Path('build/p11')
sources = {'expert': screen / 'micro', 'attention': screen / 'scan'}
if args.period == 'overlap':
    sources = {'expert': work / 'occupied/screen/micro', 'attention': work / 'finished-scan'}
tz = ZoneInfo('America/New_York')
freed = datetime.datetime(2026, 11, 27, 3, 30, 22, tzinfo=tz)
started = datetime.datetime(2026, 11, 27, 2, 2, 47, tzinfo=tz)
output = screen / f'{args.period}-operator-samples.tar.gz'
temp = output.with_suffix('.tmp')
index = io.StringIO()
writer = csv.writer(index, delimiter='\t', lineterminator='\n')
writer.writerow(('sample', 'completed_est', 'sha256'))
samples = {}
with temp.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream, tarfile.open(fileobj=stream, mode='w') as archive:
    def add(name, data, timestamp=0):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(timestamp)
        archive.addfile(info, io.BytesIO(data))

    for op, src in sources.items():
        for path in sorted(src.glob('*.tsv')):
            data = path.read_bytes()
            stamp = path.stat().st_mtime
            when = datetime.datetime.fromtimestamp(stamp, tz)
            assert (started <= when < freed if args.period == 'overlap' else when >= freed), (path, when)
            values = data.split()
            assert len(values) == (13 if op == 'expert' else 12) and re.fullmatch(b'[0-9a-f]{64}', values[-1]), path
            assert all(float(value) > 0 for value in values[-10:-1]), path
            name = f'{op}/{path.name}'
            digest = hashlib.sha256(data).hexdigest()
            writer.writerow((name, when.isoformat(), digest))
            samples[name] = digest
            add(name, data, stamp)
    add('index.tsv', index.getvalue().encode())
with tarfile.open(temp) as archive:
    for name, digest in samples.items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
if args.period == 'overlap':
    assert len(samples) == 473 + 291
temp.replace(output)
print(f'{len(samples)} {args.period} samples archived and re-read.')
