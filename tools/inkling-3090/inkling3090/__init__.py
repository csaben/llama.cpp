"""inkling-3090: one command to run Inkling 2 Mini fast on an RTX 3090.

    uvx --from "git+https://github.com/csaben/llama.cpp@inkling-3090-v1#subdirectory=tools/inkling-3090" inkling-3090

Downloads (once, cached under ~/.cache/inkling-3090):
  - a prebuilt llama-server from the csaben/llama.cpp `inkling-3090` fork (Linux x86-64, CUDA 12.8, sm_86),
  - the CUDA runtime and cuBLAS as pip wheels (pulled in by uvx; no CUDA toolkit needed),
  - the 2-bit-expert GGUF from Hugging Face,
then starts an OpenAI-compatible server on http://127.0.0.1:8031/v1 with the tuned flags.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASE_TAG = "inkling-3090-v1"
ASSET = "llama-server-inkling-linux-x64-cuda12.8-sm86.tar.gz"
ASSET_SHA256 = "92344638f9702913033e4f71fc07f47a958b27fdf755b12c742c067404401238"
ASSET_URL = f"https://github.com/csaben/llama.cpp/releases/download/{RELEASE_TAG}/{ASSET}"

MODELS = {
    False: ("arelius/Inkling-2-Mini-Q2-3090-GGUF", "Inkling-2-Mini-Q2K-Q2_0.gguf"),
    True: ("arelius/Inkling-2-Mini-Uncensored-Q2-3090-GGUF", "Inkling-2-Mini-Uncensored-Q2K-Q2_0.gguf"),
}
# the fused-kernel switches the measured numbers were taken with (see INKLING.md in the fork)
TUNED_ENV = {
    "LLAMA_INKLING_CTRL_F16": "1",
    "IK_BATCH_MESSAGES": "1",
    "IK_SEGMENTED_IQ4": "1",
    "IK_SEGMENTED_Q6": "1",
    "IK_MIX_SPECIALIZE": "1",
    "IK_CTRL_APPLY8": "1",
}
CACHE = Path(os.environ.get("INKLING3090_CACHE", Path.home() / ".cache" / "inkling-3090"))


def say(msg: str) -> None:
    print(f"[inkling-3090] {msg}", file=sys.stderr, flush=True)


def check_gpu(index: int) -> None:
    if not shutil.which("nvidia-smi"):
        sys.exit("[inkling-3090] nvidia-smi not found: an NVIDIA driver is required.")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    rows = [[c.strip() for c in line.split(",")] for line in out]
    row = next((r for r in rows if r[0] == str(index)), None)
    if row is None:
        sys.exit(f"[inkling-3090] no GPU with index {index}; found: {', '.join(r[0] + ' ' + r[1] for r in rows)}")
    _, name, cc, total, used, driver = row
    say(f"GPU {index}: {name}, compute {cc}, {int(total) - int(used)} MiB free, driver {driver}")
    if cc != "8.6":
        say("warning: the kernels are tuned for compute 8.6 (RTX 3090/3080/A6000-class); other GPUs are untested.")
    if int(total) - int(used) < 13000:
        say("warning: about 13 GB of free VRAM is needed at 64K context; lower it with --ctx if loading fails.")
    if int(driver.split(".")[0]) < 570:
        sys.exit("[inkling-3090] NVIDIA driver 570 or newer is required for the CUDA 12.8 runtime.")


def fetch_server() -> Path:
    dest = CACHE / RELEASE_TAG
    server = dest / "llama-server"
    if server.exists():
        return server
    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / ASSET
    src = os.environ.get("INKLING3090_ASSET", ASSET_URL)  # a local tarball path or URL, for testing
    say(f"downloading {src}")
    if os.path.exists(src):
        shutil.copy(src, tgz)
    else:
        urllib.request.urlretrieve(src, tgz)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if not ASSET_SHA256.startswith("@") and digest != ASSET_SHA256:
        tgz.unlink()
        sys.exit(f"[inkling-3090] checksum mismatch for {ASSET}: got {digest}")
    with tarfile.open(tgz) as t:
        t.extractall(dest, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest)
    tgz.unlink()
    server.chmod(0o755)
    return server


def cuda_lib_dirs() -> list[str]:
    dirs = []
    for mod in ("nvidia.cuda_runtime", "nvidia.cublas"):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
            if lib.is_dir():
                dirs.append(str(lib))
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(prog="inkling-3090", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uncensored", action="store_true", help="use AtomicChat's uncensored build (same speed)")
    ap.add_argument("--model", help="path to a local GGUF instead of downloading one")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    ap.add_argument("--ctx", type=int, default=65536, help="context length (default 65536)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--opencode", action="store_true", help="print an OpenCode provider config and exit")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra llama-server arguments after --")
    a = ap.parse_args()

    if a.opencode:
        # the provider block tested with OpenCode 1.2.5 (perf/opencode-inkling.json in the fork)
        import json
        print(json.dumps({"provider": {"localinkling": {
            "npm": "@ai-sdk/openai-compatible", "name": "Local Inkling",
            "options": {"baseURL": f"http://{a.host}:{a.port}/v1", "apiKey": "local"},
            "models": {"inkling-mini": {"name": "Inkling 2 Mini (Q2, 3090)",
                                        "limit": {"context": a.ctx, "output": 4096}, "tool_call": True}}}}}, indent=2))
        return

    check_gpu(a.gpu)
    server = fetch_server()
    if a.model:
        model = a.model
    else:
        from huggingface_hub import hf_hub_download
        repo, fname = MODELS[a.uncensored]
        say(f"model: {repo}/{fname} (9.9 GB, cached after the first run)")
        model = hf_hub_download(repo, fname)

    env = dict(os.environ)
    env.update(TUNED_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_lib_dirs() + [str(server.parent), env.get("LD_LIBRARY_PATH", "")])
    cmd = [str(server), "-m", model, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", str(a.ctx),
           "-np", "1", "--backend-sampling", "--host", a.host, "--port", str(a.port), "--alias", "inkling-mini"]
    cmd += [x for x in a.extra if x != "--"]
    say(f"serving on http://{a.host}:{a.port}/v1  (model id: inkling-mini; Ctrl-C to stop)")
    os.execvpe(cmd[0], cmd, env)
