#!/usr/bin/env python3
# Speculative-decoding probe (phase 5): the same prompts of three kinds (new code, an edit of given code, prose) plus an
# edit deep in a 16K-token context, through a fresh server per configuration. Prints decode tok/s and, for a drafter,
# accepted/proposed draft tokens. Temperature 0, seed 1, prompt cache off, 384 tokens per reply.
# usage: MODEL=... GPU=1 perf/spec_probe.py <label> [--no-long] [-- extra llama-server args]
import glob, json, os, subprocess, sys, time, urllib.request
args = sys.argv[1:]
label = args[0]
extra = args[args.index("--") + 1:] if "--" in args else []
no_long = "--no-long" in args
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:8041"
SNIP = open(os.path.join(ROOT, "perf", "bench.py")).read() + "\n" + open(os.path.join(ROOT, "perf", "kernsum.py")).read()
PROMPTS = [
    ("code",  "Write a Python module that implements an LRU cache with per-entry time-to-live, thread safety and hit/miss statistics. Include unit tests."),
    ("edit",  "Rewrite the following Python code so that every function and variable has type hints and the HTTP calls go through one helper function. Output the complete file, changing nothing else.\n\n```python\n" + SNIP + "\n```"),
    ("prose", "Write a 300-word story about a lighthouse keeper who finds a message in a bottle."),
]
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
M = os.path.expanduser(os.environ["MODEL"])
cmd = [os.path.join(ROOT, "build/bin/llama-server"), "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "32768",
       "-np", "1", "--port", "8041", "--host", "127.0.0.1"] + extra
proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "1")), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit("server exited early")
        try:
            urllib.request.urlopen(URL + "/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    if not no_long:
        files = sorted(glob.glob(os.path.join(ROOT, "src", "llama-*.cpp")))
        text = "".join(f"\n// ===== {os.path.basename(f)} =====\n" + open(f, errors="ignore").read() for f in files)
        toks = post("/tokenize", {"content": text})["tokens"]
        doc = post("/detokenize", {"tokens": toks[:16000]})["content"]
        PROMPTS.append(("edit@16K", doc + "\n\nCopy the first function defined in the source above exactly, then add a one-line comment above each of its statements. Output only the code."))
    for kind, p in PROMPTS:
        body = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 384, "temperature": 0, "seed": 1, "cache_prompt": False}
        t = post("/v1/chat/completions", body)["timings"]
        acc = f"  accepted/proposed {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
        print(f"{label:14s} {kind:9s} prompt {t['prompt_n']:6d}  {t['predicted_n']:4d} tok  {t['predicted_per_second']:6.1f} tok/s{acc}", flush=True)
finally:
    proc.terminate(); proc.wait()
