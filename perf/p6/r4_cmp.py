#!/usr/bin/env python3
"""Compare canonical full messages. SPEC_KINDS can require an explicit set of prompts."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("reference")
parser.add_argument("labels", nargs="+")
args = parser.parse_args()
base = args.directory
kinds = os.environ.get("SPEC_KINDS")
if kinds:
    kinds = kinds.split(",")
else:
    prefix = args.reference + "."
    kinds = sorted(p.name[len(prefix):-5] for p in base.glob(prefix + "*.json") if not p.name.endswith(".probs.json"))
if not kinds:
    parser.error("no reference messages found")


def read(label, kind):
    path = base / f"{label}.{kind}.json"
    data = path.read_bytes()
    msg = json.loads(data)
    if not any(isinstance(v, str) and v for k, v in msg.items() if k != "role"):
        raise ValueError(f"empty message in {path}")
    return data, msg


bad = False
for kind in kinds:
    ref, expected = read(args.reference, kind)
    for label in args.labels:
        data, actual = read(label, kind)
        ids_ref = base / f"{args.reference}.{kind}.ids"
        if ids_ref.exists() or os.environ.get("SPEC_REQUIRE_IDS"):
            ids = (base / f"{label}.{kind}.ids").read_bytes()
            want = ids_ref.read_bytes()
            if not want or len(want) % 4 or want != ids:
                print(f"{label:12s} {kind:9s} different or empty token IDs")
                bad = True
        if ref == data:
            print(f"{label:12s} {kind:9s} identical ({len(data)} bytes)")
            continue
        bad = True
        for key in sorted(expected.keys() | actual.keys()):
            a, b = expected.get(key), actual.get(key)
            if a != b and isinstance(a, str) and isinstance(b, str):
                pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"{label:12s} {kind:9s} {key} differs at char {pos}")
        print(f"{label:12s} {kind:9s} different JSON bytes")
raise SystemExit(int(bad))
