"""Measure one fixed-length edit at the end of a slot using the Phase 10 fixture."""
import json
import struct
import sys
import urllib.request
from pathlib import Path

depth, budget = map(int, sys.argv[1:3])
out = Path(sys.argv[3])
cache = Path("build/p11a") / f"prompt-{depth}.json"

def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:18502" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)

if not cache.exists():
    source = "".join(f"\n// ===== {p.name} =====\n" + p.read_text() for p in sorted(Path("src").glob("llama-*.cpp")))
    tokens = post("/tokenize", {"content": source})["tokens"]
    length = depth - 64
    for attempt in range(12):
        assert 0 < length <= len(tokens)
        prefix = post("/detokenize", {"tokens": tokens[:length]})["content"]
        text = prefix + "\n\nCopy the first function defined in the source above exactly, then add a one-line comment above each of its statements. Output only the code."
        messages = [{"role": "user", "content": text}]
        formatted = post("/apply-template", {"messages": messages})["prompt"]
        actual = len(post("/tokenize", {"content": formatted, "add_special": True, "parse_special": True})["tokens"])
        if actual == depth:
            break
        length += depth - actual
    assert abs(depth - actual) <= 2
    cache.write_text(json.dumps(messages))

body = dict(model="inkling-mini", messages=json.loads(cache.read_text()), max_tokens=budget, temperature=0, seed=1,
            cache_prompt=False, ignore_eos=True, verbose=True, return_tokens=True)
response = post("/v1/chat/completions", body)
msg, ids, times = response["choices"][0]["message"], response["__verbose"]["tokens"], response["timings"]
assert len(ids) == times["predicted_n"] == budget and abs(times["prompt_n"] - depth) <= 2
assert any(msg.get(k) for k in ("content", "reasoning_content", "tool_calls"))
out.with_suffix(".ids").write_bytes(struct.pack(f"<{budget}i", *ids))
out.with_suffix(".json").write_text(json.dumps(msg, sort_keys=True))
out.with_suffix(".timings").write_text(json.dumps(times, indent=2) + "\n")
