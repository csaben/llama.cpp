"""Report paired operator timings with exact output digests."""
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
suite = sys.argv[2]
variant = sys.argv[3] if len(sys.argv) > 3 else ""
legs = ("old1", "new1", "new2", "old2")
prefix = f"{suite}-{variant}" if variant else f"{suite}-"
print("case\told1_us\tnew1_us\tnew2_us\told2_us\told_mean_us\tnew_mean_us\tdelta_pct\tbit_equal")
for base in sorted(root.glob(prefix + "old1-*.tsv")):
    shape = base.stem.removeprefix(prefix + "old1-")
    samples = [(root / (prefix + leg + "-" + shape + ".tsv")).read_text().split() for leg in legs]
    times = [statistics.median(map(float, sample[3:-1])) for sample in samples]
    old = statistics.mean([times[0], times[3]])
    new = statistics.mean(times[1:3])
    same = len({sample[-1] for sample in samples}) == 1
    print(shape, *(f"{t:.3f}" for t in (*times, old, new, 100*(new/old-1))), int(same), sep="\t")
