"""Summarize the last 119 complete depth-zero head-to-head intervals of a node trace."""
import argparse
import collections
import sqlite3
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("trace")
args = parser.parse_args()
db = sqlite3.connect(args.trace)
rows = db.execute("""
    SELECT s.value, k.gridX, k.gridY, k.start, k.end
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    ORDER BY k.start
""").fetchall()
heads = [r[4] for r in rows if r[:3] == ("mul_mat_vec_q", 100512, 1)]
if len(heads) < 120:
    raise ValueError(f"only {len(heads)} head launches")
lo, hi = heads[-120], heads[-1]
rows = [r for r in rows if lo <= r[3] and r[4] <= hi]
work = collections.defaultdict(list)
glue = [r for r in rows if r[0] != "ik_gap_prefetch"]
overlap = collections.defaultdict(list)
cursor = 0
for name, nx, ny, start, end in rows:
    key = f"{name}[{nx},{ny}]" if name == "mul_mat_vec_q" else name
    work[key].append(end - start)
    if name == "ik_gap_prefetch":
        while cursor < len(glue) and glue[cursor][4] <= start:
            cursor += 1
        j = cursor
        while j < len(glue) and glue[j][3] < end:
            tag, _, _, t0, t1 = glue[j]
            overlap[tag].append(min(t1, end) - max(t0, start))
            j += 1
print("kernel, calls/token, us/call, ms/token")
for name, durations in sorted(work.items(), key=lambda pair: -sum(pair[1])):
    print(f"{name}, {len(durations)/119:.2f}, {statistics.mean(durations)/1000:.3f}, {sum(durations)/119e6:.4f}")
print(f"head-to-head ms/token: {(hi-lo)/119e6:.4f}")
for name, spans in sorted(overlap.items(), key=lambda pair: -sum(pair[1])):
    print(f"prefetch overlap with {name}, {sum(spans)/119e6:.4f} ms/token")
for a, b in [
    ("situ_q8_1_kernel", "ik_q2_down_rows"),
    ("mul_mat_vec_q", "shortconv_res_f32"),
    ("inkling_attn2_kernel", "mul_mat_vec_q"),
]:
    spans = [right[4] - left[3] for left, right in zip(glue, glue[1:]) if left[0] == a and right[0] == b]
    if spans:
        print(f"{a} to {b} inclusive, {len(spans)/119:.2f} pairs/token, {statistics.mean(spans)/1000:.3f} us/pair")
