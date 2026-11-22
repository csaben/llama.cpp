#!/usr/bin/env python3
# Speculative-decoding probe (phase 5): the same prompts of three kinds (new code, an edit of given code, prose) plus an
# edit deep in a 16K-token context, through a fresh server per configuration. Prints decode tok/s and, for a drafter,
# accepted/proposed draft tokens. Temperature 0, seed 1, prompt cache off, SPEC_MAXT (default 384) tokens per reply.
# usage: MODEL=... GPU=1 [SPEC_OUT=dir] perf/spec_probe.py <label> [--no-long] [-- extra llama-server args]
# SPEC_OUT keeps the FULL message of every reply as <label>.<kind>.json (all fields: Inkling reasons first, so a short
# reply is entirely in reasoning_content and content is empty; rounds 1-3 saved only content and compared empty files).
# Compare with perf/p6/r4_cmp.py, which refuses empty messages. SPEC_KINDS=code,edit limits the prompts,
# SPEC_REPS=n repeats each request (median rate; all n messages must be equal), SPEC_PROBS=1 also saves the tokens with
# their top-3 log-probabilities as <label>.<kind>.probs.json, SPEC_CTX sets -c (default 32768).
# SPEC_SERVER_LOG saves stderr from the server for CUDA timing or LLAMA_SPEC_PROF.
# SPEC_PROMPTS_FILE adds a JSON object of named prompts. SPEC_SERVER_BIN selects a staged server binary.
# SPEC_TOKEN_IDS=1 saves the emitted IDs even for requests that stop naturally (tool grammars require EOS).
import glob, json, os, struct, subprocess, sys, time, urllib.request
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
if os.environ.get("SPEC_PROMPTS_FILE"):
    with open(os.environ["SPEC_PROMPTS_FILE"]) as source:
        PROMPTS.extend(json.load(source).items())

def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode(errors='replace')}") from error
M = os.path.expanduser(os.environ["MODEL"])
cmd = [os.environ.get("SPEC_SERVER_BIN", os.path.join(ROOT, "build/bin/llama-server")), "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", os.environ.get("SPEC_CTX", "32768"),
       "-np", "1", "--port", "8041", "--host", "127.0.0.1"] + extra
with open(os.environ.get("SPEC_SERVER_LOG", os.devnull), "wb") as log:
    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "1")), stdout=subprocess.DEVNULL, stderr=log)
try:
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit("server exited early")
        try:
            urllib.request.urlopen(URL + "/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    kinds = os.environ.get("SPEC_KINDS", "code,edit,prose,edit@16K").split(",")
    depths = [n for n in (16, 32) if f"edit@{n}K" in kinds]
    if depths and not no_long:
        files = sorted(glob.glob(os.path.join(ROOT, "src", "llama-*.cpp")))
        text = "".join(f"\n// ===== {os.path.basename(f)} =====\n" + open(f, errors="ignore").read() for f in files)
        toks = post("/tokenize", {"content": text})["tokens"]
        for depth in depths:
            doc = post("/detokenize", {"tokens": toks[:depth*1000]})["content"]
            PROMPTS.append((f"edit@{depth}K", doc + "\n\nCopy the first function defined in the source above exactly, then add a one-line comment above each of its statements. Output only the code."))
    if "read-tool" in kinds:
        PROMPTS.append(("read-tool", "Read perf/longctx.py and explain how it measures time to first token. Identify the statement that stops the timer."))
    for kind, p in PROMPTS:
        if kind not in kinds:
            continue
        body = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": int(os.environ.get("SPEC_MAXT", "384")), "temperature": 0, "seed": 1, "cache_prompt": False}
        if kind == "read-tool":
            name = "perf/longctx.py"
            with open(os.path.join(ROOT, name)) as source:
                lines = "".join(f"{i}: {line}" for i, line in enumerate(source, 1))
            body["tools"] = [{"type": "function", "function": {"name": "read", "description": "Read a file with line numbers.",
                "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}]
            body["messages"].extend([
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_read", "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"filePath": name})}}]},
                {"role": "tool", "tool_call_id": "call_read", "content": lines},
            ])
        if os.environ.get("SPEC_FIXED_LENGTH") == "1":
            body["ignore_eos"] = True
        if body.get("ignore_eos") or os.environ.get("SPEC_TOKEN_IDS") == "1":
            body.update(return_tokens=True, verbose=True)
        if os.environ.get("SPEC_PROBS"):
            body.update({"logprobs": True, "top_logprobs": 3})
        # SPEC_REPS=n: the request n times; the median run is reported, all rates are listed
        runs = [post("/v1/chat/completions", body) for _ in range(int(os.environ.get("SPEC_REPS", "1")))]
        if body.get("ignore_eos") and any(x["timings"]["predicted_n"] != body["max_tokens"] for x in runs):
            raise ValueError(f"{label} {kind}: generation stopped before the requested token count")
        runs.sort(key=lambda x: x["timings"]["predicted_per_second"])
        r = runs[len(runs)//2]
        t = r["timings"]
        if body.get("return_tokens"):
            ids = r["__verbose"]["tokens"]
            if not ids or len(ids) != t["predicted_n"] or any(x["__verbose"]["tokens"] != ids for x in runs):
                raise ValueError(f"{label} {kind}: missing or differing output token IDs")
            if os.environ.get("SPEC_OUT"):
                os.makedirs(os.environ["SPEC_OUT"], exist_ok=True)
                path = os.path.join(os.environ["SPEC_OUT"], f"{label}.{kind}.ids")
                with open(path, "wb") as f:
                    f.write(struct.pack(f"<{len(ids)}i", *ids))
        msg = r["choices"][0]["message"]
        chars = sum(len(v) for k, v in msg.items() if isinstance(v, str) and k != "role")
        if not chars:
            raise ValueError(f"{label} {kind}: empty reply")
        if any(x["choices"][0]["message"] != msg for x in runs):
            raise ValueError(f"{label} {kind}: repeats differ")
        if os.environ.get("SPEC_OUT"):  # keep the whole message, to compare drafted against plain output
            os.makedirs(os.environ["SPEC_OUT"], exist_ok=True)
            json.dump(msg, open(os.path.join(os.environ["SPEC_OUT"], f"{label}.{kind}.json"), "w"), sort_keys=True)
            if os.environ.get("SPEC_PROBS"):
                json.dump(r["choices"][0].get("logprobs"), open(os.path.join(os.environ["SPEC_OUT"], f"{label}.{kind}.probs.json"), "w"))
        acc = f"  accepted/proposed {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
        allr = "  runs " + "/".join(f"{x['timings']['predicted_per_second']:.0f}" for x in runs) if len(runs) > 1 else ""
        print(f"{label:14s} {kind:9s} prompt {t['prompt_n']:6d}  {t['predicted_n']:4d} tok  {t['predicted_per_second']:6.1f} tok/s  {chars:6d} chars{acc}{allr}", flush=True)
finally:
    proc.terminate(); proc.wait()
