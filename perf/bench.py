import json, sys, urllib.request, statistics
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8031"
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like 'P3DT4H5M' into seconds, with tests.",
    "Write a 200-word paragraph about the history of the bicycle.",
    "Explain step by step how TCP slow start and congestion avoidance work.",
]
res = []
for temp in (0.0, 0.7):
    for p in PROMPTS:
        body = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 700, "temperature": temp}
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        t = d["timings"]
        acc = f" draft {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
        res.append((temp, t["predicted_per_second"]))
        print(f"  T={temp} {t['predicted_n']:4d} tok  gen {t['predicted_per_second']:6.1f} tok/s  prompt {t['prompt_per_second']:6.1f}{acc}", flush=True)
for temp in (0.0, 0.7):
    print(f"  mean gen @T={temp}: {statistics.mean(v for tt, v in res if tt == temp):.1f} tok/s")
