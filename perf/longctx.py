#!/usr/bin/env python3
# Long-context served benchmark (phase 5): one 16K-token and one 32K-token chat prompt made of this repository's own
# source files, a 500-token reply each, temperature 0, prompt cache off. Reports prompt tokens, time to first token
# (measured at the client on the stream), prefill tok/s and decode tok/s (the server's timings).
# usage: perf/longctx.py [URL]                      against a running server (default http://127.0.0.1:8031)
#        MODEL=... GPU=1 perf/longctx.py --start [--sizes 16000,32000] [--short] [-- extra llama-server args]
#   --start launches build/bin/llama-server (64K context, Q8_0 KV, one slot) on port 8041 and stops it afterwards;
#   --short also runs perf/bench.py (short prompts) against the same server.
import glob, json, os, subprocess, sys, time, urllib.request
args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
start = "--start" in args
short = "--short" in args
sizes = [int(x) for x in args[args.index("--sizes") + 1].split(",")] if "--sizes" in args else [16000, 32000]
n_reply = int(args[args.index("--reply") + 1]) if "--reply" in args else 500
urls = [a for a in args if a.startswith("http")]
URL = urls[0] if urls else ("http://127.0.0.1:8041" if start else "http://127.0.0.1:8031")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)

def corpus():
    files = sorted(glob.glob(os.path.join(ROOT, "src", "llama-*.cpp")))
    return "".join(f"\n// ===== {os.path.basename(f)} =====\n" + open(f, errors="ignore").read() for f in files)

def run(n_target, text):
    toks = json.load(post("/tokenize", {"content": text}))["tokens"]
    doc = json.load(post("/detokenize", {"tokens": toks[:n_target]}))["content"]
    q = "\n\nThe text above is C++ source code. Write a detailed review of it: list the main classes and functions, what each does, and three concrete risks or bugs you would check first."
    body = {"model": "x", "messages": [{"role": "user", "content": doc + q}], "max_tokens": n_reply, "temperature": 0,
            "stream": True, "cache_prompt": False, "timings_per_token": True}
    t0 = time.time(); ttft = None; timings = None; n_chunks = 0
    for line in post("/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        ch = d.get("choices") or [{}]
        delta = ch[0].get("delta", {}) if ch else {}
        if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
            ttft = time.time() - t0
        if d.get("timings"):
            timings = d["timings"]
    t = timings
    acc = f"  draft {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
    print(f"  prompt {t['prompt_n']:6d} tok  TTFT {ttft:6.2f} s  prefill {t['prompt_per_second']:7.1f} tok/s  "
          f"reply {t['predicted_n']:4d} tok  decode {t['predicted_per_second']:6.1f} tok/s{acc}", flush=True)

proc = None
if start:
    M = os.path.expanduser(os.environ["MODEL"])
    cmd = [os.path.join(ROOT, "build/bin/llama-server"), "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "65536",
           "-np", "1", "--port", "8041", "--host", "127.0.0.1"] + extra
    print("server:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "1")), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(1200):
        if proc is not None and proc.poll() is not None:
            raise SystemExit("server exited early")
        try:
            urllib.request.urlopen(URL + "/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    if proc is not None:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", "-i", os.environ.get("GPU", "1")], capture_output=True, text=True).stdout.strip()
        print("GPU memory after load:", mem, flush=True)
    if short:
        subprocess.run([sys.executable, os.path.join(ROOT, "perf/bench.py"), URL], check=False)
    text = corpus()
    for n in sizes:
        run(n, text)
finally:
    if proc is not None:
        proc.terminate(); proc.wait()
