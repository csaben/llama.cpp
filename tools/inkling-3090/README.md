# inkling-3090

Run Inkling 2 Mini at about 200 tok/s on one RTX 3090. One command, with no CUDA toolkit and no build:

```bash
uvx --from "git+https://github.com/csaben/llama.cpp@inkling-3090-v1#subdirectory=tools/inkling-3090" inkling-3090
```

The first run downloads the following once and caches it under `~/.cache/inkling-3090` and the Hugging Face cache:

- A prebuilt `llama-server` from this fork, for Linux x86-64 with CUDA 12.8 and sm_86.
- The CUDA runtime and cuBLAS, as pip wheels that uvx installs.
- The 2-bit-expert model, [arelius/Inkling-2-Mini-Q2-3090-GGUF](https://huggingface.co/arelius/Inkling-2-Mini-Q2-3090-GGUF) (9.9 GB).

It then serves an OpenAI-compatible API at `http://127.0.0.1:8031/v1`, with model id `inkling-mini` and 64K of context.

## Options

| Flag | Effect |
|---|---|
| `--uncensored` | Use [AtomicChat's uncensored build](https://huggingface.co/arelius/Inkling-2-Mini-Uncensored-Q2-3090-GGUF) instead. Same speed. |
| `--gpu N` | Choose the GPU (default 0). |
| `--ctx N` | Set the context length (default 65536; it needs about 13 GB of free VRAM at 64K). |
| `--port N`, `--host H` | Set where to listen (default 127.0.0.1:8031). |
| `--model PATH` | Use a local GGUF. |
| `--opencode` | Print an [OpenCode](https://opencode.ai) provider config for this server, then exit. |
| `-- <args>` | Pass the rest through to `llama-server`. |

## Requirements

- Linux on x86-64.
- An NVIDIA driver at version 570 or newer.
- A 24 GB Ampere card: RTX 3090 or 3090 Ti, or A5000/A6000-class. Other GPUs may work but are untested; the kernels are tuned for compute capability 8.6.
- [uv](https://docs.astral.sh/uv/).

## What to expect

These were measured on an RTX 3090, using this fork's numbers and an independent audit:

| Context in use | Decode tok/s |
|---|---:|
| empty | ~208 |
| 1K | ~193 |
| 16K–24K | ~141 |
| 32K–48K | ~123 |

Coding agents usually start with 16K+ tokens of tool prompt, so expect 120–140 tok/s there. Quality is about 3–4%
higher perplexity than the 4-bit conversion. For quality-first use, build the fork yourself (see
[INKLING.md](../../INKLING.md)) and run ggml-org's IQ4_NL GGUF.

The code is MIT, like llama.cpp. The models are Apache-2.0; see their cards.
