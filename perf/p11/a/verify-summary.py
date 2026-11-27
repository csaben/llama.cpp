"""Recheck the retained comparison evidence before selecting a runtime."""
import hashlib
import re
import sys
from pathlib import Path

root = Path(__file__).parent
checks = root / "checks"
binary = Path("build/p11a/checks")
name = sys.argv[1]

def contents(label):
    return (checks / f"{label}.txt").read_text()

def equal_logits(label, nrows):
    current = binary / f"{name}-{label}.log.bin"
    control = binary / f"p9-{label}.log.bin"
    for path in (current, control):
        assert path.stat().st_size == nrows*200058*4, path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == contents(path.name.removesuffix(".log.bin")).split()[0], path
    assert current.read_bytes() == control.read_bytes(), current

for mode in (0, 1):
    equal_logits(f"default-mixed-{mode}", 64)
    for prefix in (1, 4096, 16384, 32768, 65528):
        equal_logits(f"logits-{mode}-{prefix}", 8)
    for prefix in (65536, 128064):
        equal_logits(f"deep-{mode}-{prefix}", 8)
print("Default one-card flags: 128 interleaved full-vocabulary rows matched p9 bit for bit.")
print("Two-card output: 112 full-vocabulary rows matched p9 bit for bit over seven depths through 128,064 and both flag settings.")
for variant in ("uncensored", "calibrated"):
    for mode in ("pp", "tg", "tg-mr"):
        def ppl(version):
            text = contents(f"{version}-{variant}-{mode}")
            return float(re.search(r"Final estimate: PPL = ([0-9.]+)", text)[1])
        before, after = ppl("p9"), ppl(name)
        assert abs(after/before - 1) <= 0.001
        print(f"{variant} {mode}: p9 {before:.4f}, {name} {after:.4f}, difference {100*(after/before-1):+.4f}%")
labels = ["trace", "edge-1533", "edge-3069", "edge-4500", "control"] + [f"rand-{p}" for p in (2000, 2046, 2558, 3069, 4094, 4500)]
def count(label):
    match = re.search(r"(\d+) (?:output )?rows (?:up to position|in \d+ steps).*?: 0 rows differ from single-token decode, max \|logit diff\| 0$", contents(f"{name}-{label}").strip())
    assert match, label
    return int(match[1])
assert sum(map(count, labels)) == 2714
print(f"Rollback and boundary battery: 2,714 exact rows in eleven checks. Extra widths 2-7: {count('widths')} exact rows.")
for key in ("head", "inkling"):
    pairs = re.findall(r"(\d+)/(\d+) tests passed", contents(f"{name}-ops-{key}"))
    assert len(pairs) == 1 and int(pairs[0][0]) > 0 and pairs[0][0] == pairs[0][1], pairs
    print(f"test-backend-ops {key}: {pairs[0][0]}/{pairs[0][1]}.")
