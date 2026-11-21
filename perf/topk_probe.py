#!/usr/bin/env python3
# start llama-server, ask for the top-5 logprobs of the first generated token of each greedy prompt, print them
# usage: perf/topk_probe.py [env=VAL ...]   (e.g. LLAMA_INKLING_FUSED=0)
import json, os, subprocess, sys, time, urllib.request
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
for a in sys.argv[1:]:
    k, v = a.split("=", 1); env[k] = v
M = os.path.expanduser("~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf")
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like 'P3DT4H5M' into seconds, with tests.",
    "The history of the bicycle begins in the early nineteenth century, when",
    "Explain step by step how TCP slow start and congestion avoidance work.",
]
port = 8039
p = subprocess.Popen(["build/bin/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-c", "4096", "--port", str(port), "--host", "127.0.0.1", "-np", "1"],
                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    for pr in PROMPTS:
        body = {"prompt": pr, "n_predict": 1, "temperature": 0, "n_probs": 5, "samplers": []}
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        probs = d["completion_probabilities"][0]["top_logprobs"] if "completion_probabilities" in d else d
        print(f"{pr[:50]!r}: " + "  ".join(f"{t['token']!r}:{t['logprob']:.4f}" for t in probs))
finally:
    p.terminate(); p.wait()
