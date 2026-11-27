"""Compare every public CMake cache entry after normalizing the source worktree."""
import hashlib
import json
import re
from pathlib import Path

root = Path.cwd()
ref = Path.home() / 'llama.cpp-fast'
pattern = re.compile(r'^([^/#\s][^:\n]*):([^=\n]+)=(.*)$', re.M)


def read_cache(path):
    entries = pattern.findall((path / 'build/CMakeCache.txt').read_text())
    values = {key: (kind, value.replace(str(path) + '/', '<worktree>/'))
              for key, kind, value in entries if kind not in ('INTERNAL', 'STATIC')}
    generator = next(value for key, _, value in entries if key == 'CMAKE_GENERATOR')
    return values, generator


actual, generator = read_cache(root)
expected, ref_generator = read_cache(ref)
assert generator == ref_generator
assert actual == expected, {k: (expected.get(k), actual.get(k)) for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k)}
print(f'{len(actual)} public CMake cache entries have identical values and types after path normalization.')
print('Generator', generator)
for key in ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_CUDA_ARCHITECTURES', 'CMAKE_BUILD_TYPE'):
    print(key, actual[key][1])
print('Normalized cache SHA-256', hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest())
