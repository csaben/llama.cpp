"""Configure this worktree with the reference build's options and check them."""
import os
import re
import subprocess
from pathlib import Path

root = Path.cwd()
ref = Path.home() / 'llama.cpp-fast'
pattern = re.compile(r'^([\w-]+):(\w+)=(.*)$', re.M)


def options(cache):
    return {name: (kind, value) for name, kind, value in pattern.findall(cache.read_text())
            if (kind in ('BOOL', 'STRING', 'UNINITIALIZED') and name.startswith(('LLAMA_', 'GGML_', 'BUILD_', 'CMAKE_')))
            or name in ('CUDAToolkit_ROOT', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER')}


cache = ref / 'build/CMakeCache.txt'
generator = re.search(r'^CMAKE_GENERATOR:INTERNAL=(.+)$', cache.read_text(), re.M)[1]
requested = {name: (kind, value.replace(str(ref) + '/', str(root) + '/')) for name, (kind, value) in options(cache).items()}
command = ['cmake', '-S', '.', '-B', 'build', '-G', generator, *(f'-D{key}:{kind}={val}' for key, (kind, val) in requested.items())]
if __name__ == '__main__':
    if os.environ.get('P11_CHECK_CONFIG') != '1':
        subprocess.run(['perf/p11/b/gpu.sh', *command], check=True)
    actual = options(root / 'build/CMakeCache.txt')
    assert actual == requested, {k: (requested.get(k), actual.get(k)) for k in requested.keys() | actual.keys()
                                 if requested.get(k) != actual.get(k)}
    print(f'{len(requested)} build options and compiler paths match {cache}')
