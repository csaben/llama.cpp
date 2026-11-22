"""Measure overlap between the two CUDA streams of the frozen-input replay."""
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
rows = db.execute("""
    SELECT k.start, k.end, k.streamId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName=s.id ORDER BY k.start
""").fetchall()
starts = [a for a, _, _, name in rows if name == "ik_replay_inputs"]
if len(starts) < 120:
    raise ValueError("expected at least 120 captured tokens")
spans = []
for lo, hi in zip(starts[-120:-1], starts[-119:]):
    kernels = [(a, b, stream, name) for a, b, stream, name in rows if lo <= a < hi and name != "ik_replay_inputs"]
    streams = {stream for _, _, stream, _ in kernels}
    if len(streams) != 2:
        raise ValueError(f"expected two streams, got {streams}")
    matstream = next(stream for _, _, stream, name in kernels if name == "ik_q2_down_rows")
    mat = [(a, b) for a, b, stream, _ in kernels if stream == matstream]
    glue = [(a, b) for a, b, stream, _ in kernels if stream != matstream]
    i = j = overlap = 0
    while i < len(mat) and j < len(glue):
        x, y = mat[i]
        u, v = glue[j]
        overlap += max(0, min(y, v) - max(x, u))
        if y < v:
            i += 1
        else:
            j += 1
    spans.append([len(mat), len(glue), sum(b-a for a,b in mat)/1e6, sum(b-a for a,b in glue)/1e6, overlap/1e6, (hi-lo)/1e6])
print("mean of 119 warmed token intervals")
for name, series in zip(("matvec calls", "glue calls", "matvec busy ms", "glue busy ms", "concurrent ms", "interval ms"), zip(*spans)):
    print(f"{name}: {statistics.mean(series):.4f}")
