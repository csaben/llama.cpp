"""Keep the kernel launches and CUDA API names used by the Stream B Nsight analyses."""
import csv
import lzma
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

source, summary, archive = map(Path, sys.argv[1:4])
with tempfile.TemporaryDirectory() as tmp:
    database = Path(tmp) / 'kernels.sqlite'
    with sqlite3.connect(database) as con:
        con.execute('ATTACH DATABASE ? AS captured', (f'file:{source.resolve()}?mode=ro',))
        con.execute('''CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL AS
            SELECT start,end,deviceId,streamId,correlationId,demangledName,gridX,gridY,gridZ,
                   blockX,blockY,blockZ,registersPerThread,staticSharedMemory,dynamicSharedMemory
            FROM captured.CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start''')
        con.execute('''CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME AS
            SELECT correlationId,nameId FROM captured.CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL)''')
        con.execute('''CREATE TABLE StringIds AS SELECT id,value FROM captured.StringIds
            WHERE id IN (SELECT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL UNION SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME)''')
        totals = {name: (calls,ns) for name,calls,ns in con.execute('''
            SELECT value, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL
            JOIN StringIds ON demangledName=id GROUP BY value''')}
    with summary.open() as f:
        expected = {r['Name']: (int(r['Instances']), int(r['Total Time (ns)'])) for r in csv.DictReader(f)}
    assert totals == expected, source
    with database.open('rb') as src, lzma.open(archive, 'wb', preset=6) as dst:
        shutil.copyfileobj(src, dst)
    print(f'{archive.name}: {sum(n for n,_ in totals.values())} launches, {len(totals)} kernel names, all counts and durations match Nsight')
