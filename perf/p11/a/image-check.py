"""Compare greedy token IDs and the complete assistant message for a real image."""
import base64
import json
import mimetypes
import struct
import sys
import urllib.request
from pathlib import Path

image, budget, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
mime = mimetypes.guess_type(image.name)[0]
assert mime and mime.startswith('image/')
url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")
content = [
    {"type": "text", "text": "Describe the objects and their colors in this photograph."},
    {"type": "image_url", "image_url": {"url": url}},
]
body = dict(model="inkling-mini", messages=[dict(role="user", content=content)], max_tokens=budget,
            temperature=0, seed=1, cache_prompt=False, ignore_eos=True, verbose=True, return_tokens=True)
req = urllib.request.Request("http://127.0.0.1:18502/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as handle:
    reply = json.load(handle)
msg, ids, times = reply["choices"][0]["message"], reply["__verbose"]["tokens"], reply["timings"]
assert len(ids) == times["predicted_n"] == budget and times["prompt_n"] > 32
assert any(msg.get(key) for key in ("content", "reasoning_content", "tool_calls"))
out.with_suffix(".ids").write_bytes(struct.pack(f"<{budget}i", *ids))
out.with_suffix(".json").write_text(json.dumps(msg, sort_keys=True))
out.with_suffix(".timings").write_text(json.dumps(times, indent=2) + "\n")
