#!/usr/bin/env python3
# For each genN.txt that differs between two output directories of perf/server_run.py (or perf/greedy.sh), ask a fresh
# server for the top-5 logprobs of the next token after the common prefix. A divergence counts as a near-tie flip when
# the top two logprobs are within ~0.1.
# usage: MODEL=... GPU=1 perf/margin_probe.py <dir_a> <dir_b> [env=VAL ...] [-- extra server args]
import json, os, subprocess, sys, time, urllib.request
args = sys.argv[1:]
a, b = args[0], args[1]
rest = args[2:]
extra = rest[rest.index("--") + 1:] if "--" in rest else []
envs = rest[:rest.index("--")] if "--" in rest else rest
env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "1"))
for e in envs:
    k, v = e.split("=", 1); env[k] = v
M = os.path.expanduser(os.environ["MODEL"])
port = 8039
todo = []
for i in range(1, 10):
    fa, fb = os.path.join(a, f"gen{i}.txt"), os.path.join(b, f"gen{i}.txt")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    ta, tb = open(fa, "rb").read(), open(fb, "rb").read()
    if ta == tb:
        print(f"gen{i}: identical"); continue
    n = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
    while n > 0 and (ta[n - 1] & 0xC0) == 0x80:  # do not cut inside a UTF-8 sequence
        n -= 1
    todo.append((i, n, ta[:n].decode("utf-8", "ignore"), ta[n:n + 24], tb[n:n + 24]))
if todo:
    p = subprocess.Popen(["build/bin/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384",
                          "--port", str(port), "--host", "127.0.0.1", "-np", "1"] + extra, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception:
                time.sleep(0.5)
        for i, n, prefix, ca, cb in todo:
            # back off to a whitespace boundary so the prefix tokenizes the way it was generated
            cut = max(prefix.rfind(" "), prefix.rfind("\n"))
            body = {"prompt": prefix[:cut], "n_predict": 8, "temperature": 0, "n_probs": 5, "samplers": ["top_k"], "top_k": 1, "cache_prompt": False}
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            print(f"gen{i}: first difference at byte {n}: {ca!r} vs {cb!r}; next tokens after byte {cut}:")
            for t in d["completion_probabilities"]:
                top = t["top_logprobs"]
                print(f"   {top[0]['token']!r}: " + "  ".join(f"{x['token']!r}:{x['logprob']:.4f}" for x in top[:3]) + f"   margin {top[0]['logprob'] - top[1]['logprob']:.4f}")
    finally:
        p.terminate(); p.wait()
