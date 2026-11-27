"""Check completed replays against p9 plain and retain one row per real request."""
import json
import sys
from pathlib import Path

root = Path(__file__).parent / "capacity"
pairs = []
for slot in (139264, 141312, 143360, 145408, 147456):
    ref = f"after-p9-ref-{slot}"
    pairs.append((slot, f"after-p9-repeat-{slot}", ref))
    if slot in (139264, 147456):
        pairs.append((slot, f"after-direct-load-{slot}", ref))
    if slot in (141312, 143360, 145408):
        pairs.append((slot, f"after-release-fit-{slot}", ref))
for label in ("after-release-olda-128", "after-direct-shared-128"):
    pairs.append((131072, label, "p9-128-plain"))
for variant in ("layer", "table"):
    pairs.append((147456, f"after-direct-{variant}-147456", "after-p9-ref-147456"))
pairs.extend([(196608, "after-p9-deep-196608", "after-p9-ref-196608"),
              (131072, "after-direct-real-image", "after-p9-real-image")])
missing = []
print("slot\trequest\tplain_control\tprompt_tokens\temitted_ids\tfull_reply_and_ids")
for slot, label, ref in pairs:
    if not all((root / f"{name}.{ext}").is_file() for name in (ref, label) for ext in ("timings", "ids", "json")):
        missing.append(label)
        continue
    a, b = [json.loads((root / f"{name}.timings").read_text()) for name in (ref, label)]
    n = 384 if label.endswith("image") else 3000
    depth = 238 if n == 384 else slot-3008
    assert a["prompt_n"] == b["prompt_n"] == depth and a["predicted_n"] == b["predicted_n"] == n
    for ext in ("ids", "json"):
        data = (root / f"{label}.{ext}").read_bytes()
        assert data == (root / f"{ref}.{ext}").read_bytes(), (label, ext)
        if ext == "ids":
            assert len(data) == n*4
    print(slot, label, ref, depth, n, "equal", sep="\t")
if missing:
    print("Pending:", ", ".join(missing), file=sys.stderr)
    if "--partial" not in sys.argv:
        sys.exit(1)
