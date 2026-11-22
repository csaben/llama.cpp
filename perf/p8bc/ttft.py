"""Measure client time to first visible or reasoning token after an uncached file read."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def post(path, body):
    request = urllib.request.Request(
        'http://127.0.0.1:18432' + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=600)


def run(doc, limit, expected):
    messages = [
        {"role": "user", "content": "Review the implementation in the file bundle."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "read_source", "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"filePath": "src/bundle.cpp"})}}]},
        {"role": "tool", "tool_call_id": "read_source", "content": doc},
        {"role": "user", "content": "Summarize the first function and the data structures it uses."},
    ]
    body = {"model": "x", "messages": messages, "max_tokens": limit, "temperature": 0, "seed": 1,
            "cache_prompt": False, "stream": True, "timings_per_token": True}
    start = time.monotonic()
    first = timing = None
    with post('/v1/chat/completions', body) as stream:
        for line in stream:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            delta = (event.get('choices') or [{}])[0].get('delta', {})
            if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls')):
                first = time.monotonic() - start
            if event.get('timings'):
                timing = event['timings']
    assert first is not None and timing and timing['prompt_n'] >= expected and timing['predicted_n'] > 0
    print(timing['prompt_n'], round(first, 4), round(timing['prompt_per_second'], 2),
          timing['predicted_n'], round(timing['predicted_per_second'], 2), flush=True)


def main():
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'p8b-final', 'src'], text=True)
    paths = sorted(Path(p) for p in files.splitlines() if Path(p).parent == Path('src') and Path(p).match('llama-*.cpp'))
    text = ''.join(f'\n// ===== {p.name} =====\n' + subprocess.check_output(
        ['git', 'show', f'p8b-final:{p}'], text=True) for p in paths)
    with post('/tokenize', {'content': text}) as response:
        tokens = json.load(response)['tokens']
    docs = []
    for size in (4096, 16384, 32768):
        with post('/detokenize', {'tokens': tokens[:size]}) as response:
            docs.append((size, json.load(response)['content']))
    run(docs[0][1][:2000], 16, 0)
    for _ in range(3):
        for size, doc in docs:
            run(doc, 128, size)


if __name__ == '__main__':
    main()
