#!/usr/bin/env python3
# start llama-server on GPU1 with the given extra args, run the 3 greedy prompts (200 tokens, temp 0) through
# /completion and the throughput bench (perf/bench.py), save the outputs under <outdir>, stop the server
# usage: perf/server_run.py <outdir> [--no-bench] [--kv-probe] [-- extra server args...]
import json, os, subprocess, sys, time, urllib.request
args = sys.argv[1:]
outdir = args.pop(0)
no_bench = "--no-bench" in args
if no_bench:
    args.remove("--no-bench")
kv_probe = "--kv-probe" in args
if kv_probe:
    args.remove("--kv-probe")
extra = args[1:] if args and args[0] == "--" else args
os.makedirs(outdir, exist_ok=True)
M = os.path.expanduser(os.environ.get("MODEL", "~/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf"))
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration like 'P3DT4H5M' into seconds, with tests.",
    "The history of the bicycle begins in the early nineteenth century, when",
    "Explain step by step how TCP slow start and congestion avoidance work.",
]
port = 8041
if kv_probe:
    # A batched prefill followed by multiple single-token decode steps. Checking
    # only the first token tests the last prefill logit, not the fused KV reader.
    PROMPTS = ["A bicycle wheel has spokes, a hub, a rim and a tire. " * 48
               + "\nExplain in a sentence what the spokes do."]
cmd = ["build/bin/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384", "-np", "1",
       "--port", str(port), "--host", "127.0.0.1"] + extra
print("server:", " ".join(cmd), flush=True)
log = open(os.path.join(outdir, "server.log"), "w")
p = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "0")), stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(1200):
        if p.poll() is not None:
            raise SystemExit("server exited early, see server.log")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    for i, pr in enumerate(PROMPTS, 1):
        body = {"prompt": pr, "n_predict": 12 if kv_probe else 200, "temperature": 0,
                "samplers": ["top_k"], "top_k": 1, "seed": 1, "cache_prompt": False}
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        open(os.path.join(outdir, f"gen{i}.txt"), "w").write(pr + d["content"])
        t = d["timings"]
        acc = f" draft {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
        print(f"gen{i}: {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s{acc}", flush=True)
    if not no_bench:
        subprocess.run([sys.executable, "perf/bench.py", f"http://127.0.0.1:{port}"], check=False)
finally:
    p.terminate(); p.wait()
