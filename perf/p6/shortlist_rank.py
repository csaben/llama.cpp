#!/usr/bin/env python3
"""Token ranking for the DSpark drafter's shortlist head (round 3; the idea is ingine's, Apache-2.0,
https://github.com/satellitedown/inkling-2-mini-abliterated-ingine native/tools/build_draft_shortlist.py: token
frequency over local code and English text; this script and its ranking are our own, nothing was copied).

Tokenizes files through a running llama-server (/tokenize) and writes the token ids by descending count as
little-endian uint32 (tokens never seen are not listed; dspark_fold.py fills up by id).

usage: perf/p6/shortlist_rank.py http://127.0.0.1:8041 OUT.bin [MB of code, default 24] [MB of text, default 24]
Corpus: system files only (python stdlib, C headers, this repository's sources; /usr/share/doc and man pages).
"""
import collections, glob, gzip, json, os, random, struct, sys, urllib.request

url, out = sys.argv[1], sys.argv[2]
mb_code = float(sys.argv[3]) if len(sys.argv) > 3 else 24
mb_text = float(sys.argv[4]) if len(sys.argv) > 4 else 24
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = ['/usr/lib/python3*/**/*.py', '/usr/include/**/*.h', ROOT + '/src/**/*.cpp', ROOT + '/common/**/*.cpp', ROOT + '/tools/**/*.cpp',
        ROOT + '/ggml/src/**/*.c', ROOT + '/ggml/src/**/*.cpp', ROOT + '/tools/server/webui/src/**/*.ts', ROOT + '/tools/server/webui/src/**/*.svelte',
        ROOT + '/**/*.py', ROOT + '/**/*.sh', ROOT + '/**/CMakeLists.txt', '/usr/lib/node_modules/**/*.js', '/usr/share/**/*.json']
TEXT = ['/usr/share/doc/**/*.md', '/usr/share/doc/**/*.txt', '/usr/share/doc/**/README*', '/usr/share/doc/**/*.rst', '/usr/share/doc/**/*.html',
        '/usr/share/doc/**/changelog*.gz', '/usr/share/man/man[1-8]/*.gz', ROOT + '/**/*.md']

def collect(patterns, budget):
    files = sorted(set(f for p in patterns for f in glob.glob(p, recursive=True) if os.path.isfile(f)))
    random.Random(0).shuffle(files)
    texts, n = [], 0
    for f in files:
        try:
            t = (gzip.open(f, 'rt', encoding='utf-8') if f.endswith('.gz') else open(f, encoding='utf-8')).read(200000)
        except Exception:
            continue
        if t.strip():
            texts.append(t); n += len(t)
            if n > budget:
                break
    return texts, n

counts = collections.Counter()
total = 0
for name, pats, mb in (('code', CODE, mb_code), ('text', TEXT, mb_text)):
    texts, n = collect(pats, mb*1e6)
    print(f'{name}: {len(texts)} files, {n/1e6:.1f} MB', file=sys.stderr)
    buf, size = [], 0
    def flush():
        global total
        if not buf:
            return
        req = urllib.request.Request(url + '/tokenize', json.dumps({'content': '\n'.join(buf)}).encode(), {'Content-Type': 'application/json'})
        toks = json.load(urllib.request.urlopen(req, timeout=600))['tokens']
        counts.update(toks); total += len(toks)
        buf.clear()
    for t in texts:
        buf.append(t); size += len(t)
        if size > 500000:
            flush(); size = 0
    flush()
rank = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
open(out, 'wb').write(struct.pack(f'<{len(rank)}I', *rank))
cum, top = 0, {}
for i, t in enumerate(rank):
    cum += counts[t]
    if i + 1 in (4096, 16384, 32768, 49152, 65536):
        top[i + 1] = cum/total
print(f'{total} tokens, {len(rank)} distinct; coverage by rank: ' + ', '.join(f'{k}: {v:.4f}' for k, v in top.items()), file=sys.stderr)
