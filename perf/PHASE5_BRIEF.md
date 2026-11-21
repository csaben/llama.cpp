# Phase 5 brief: long-context decode and grouped-expert speculative verify

Read `perf/RESULTS.md` (phases 1–4) and `INKLING.md` first. If `perf/HANDOFF.md` exists, read it next: a previous
agent may have started this phase, and it lists what is done and what is in flight.

## Why

The calibrated 2-bit model (`~/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf`, live flags in
`perf/live/live-env.txt`) does about 208 tok/s at an empty context. The user mostly runs it inside a coding agent
(OpenCode), with a 17K-token tool prompt and contexts of 16K–60K. The server logs from 131 real requests show:

| Context | Decode tok/s |
|---|---:|
| < 4K | 196.5 |
| 4–16K | 151 |
| 16–24K | 141 |
| 24–32K | 132 |
| 32–48K | 123 |

Prompt processing on a 17K-token prompt was about 1,900 tok/s (8.9 s).

The goal is to keep the 2-bit model at about 200 tok/s at the long contexts that coding agents actually use, or get
as close as the hardware allows. Improving the near-exact builds (IQ4_NL dense, Q6_K dense) at depth is also in scope.

## Reference to study and cite

Study <https://github.com/alesha-pro/maya-inkling-mini> (Project Maya `maya-ik2` engine, MIT license), especially
`docs/INKLING2-MINI.md` ("how it works" and what didn't work). It reports:

- Only 8 of the 48 layer visits attend over the full prefix, reading about 16 KB of KV per context token.
- An FP8 KV cache, with attention reading KV at up to 846 GB/s. It holds 104 tok/s at a 64K prefix (Q4 weights).
- Speculative verification with the official DSpark drafter. Each token's expert picks are grouped on the GPU, so each
  expert is read once per window, and the whole window runs as one CUDA graph. It drafts only when that pays. This
  gives 150–210 tok/s on code and edits.
- Prefill at 2,700 tok/s on 4K tokens.

Ideas are fair game. If you copy or closely adapt any code, keep its MIT notice: add the file header and a
`NOTICE`/attribution line, and say so in RESULTS.md. For every technique you use, add the source, whether a paper,
repo, or PR, to a "Sources" list in the phase 5 section of RESULTS.md. The user wants every trick cited.

## Workstreams (profile first, then rank by measured payoff)

1. **Long-context attention.** Measure first: how decode time splits between KV reads and weights at 4K, 16K and
   32K, and what bandwidth the fused attention kernel achieves on the full-attention visits. Then attack it:
   - Cheaper KV formats. Try FP8 E4M3 (Ampere has no FP8 math; dequantize in registers) and lower-bit K or V.
     Keep the Hadamard rotation basis consistent between prefill and decode; phase 4's cache-basis bug lives here.
   - A bandwidth-optimal split-K / flash-decoding schedule at large extents.
   - Anything that avoids re-reading the full prefix on the visits that don't need it.
2. **Grouped-expert speculative verify** (DSpark, `--spec-type draft-dspark`, drafter
   `~/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf`):
   - A fused verify path for n = 2–8 tokens that reads each selected expert once for all the tokens that picked it.
   - The window runs as one CUDA graph.
   - Adaptive drafting: don't draft when the expected acceptance doesn't pay.
   Phase 2 showed that the naive verify reads n× the expert bytes, so this is the fix for that. Report acceptance and
   tok/s separately for code, edits and prose.
3. **Prefill**, if time allows: the 17K-token prompt takes 8.9 s.

## Measurement standard (record this for every accepted configuration)

The user wants a researcher to find these tables worth reading, so keep a phase 5 table in RESULTS.md with:

- **Decode:** `llama-bench ... -p 0 -n 128 -d 0,4096,16384,32768 -r 3` (tok/s ± sd) for the 2-bit model, and for
  IQ4_NL dense where relevant.
- **Prefill:** `llama-bench -p 512,4096,16384 -n 0` (tok/s).
- **Served:** `perf/bench.py` (short prompt, T=0 and 0.7). Also a long-context served run: one 16K-token and one
  32K-token prompt with a 300–700-token reply, reporting decode tok/s and time to first token. Add a script for
  this to `perf/`.
- **Quality:** prefill PPL (8×2048, `perf/ppl.txt`) and decode-path PPL (2×1024, `-ub 1`); the greedy probes
  (`perf/greedy.sh`); and the 827-token mixed-prefill regression probe from INKLING.md.
- **Speculative decoding:** acceptance (accepted/proposed) per prompt type, plus plain vs DSpark tok/s on the same
  prompts.

Quality gates: exact-math changes must keep PPL within about 0.5% of the same model's current value. Anything that
changes numerics, such as a new KV format, ships as an opt-in variant with its own PPL row and is never silently the
default.

## Rules

- **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`). **Never use GPU0**, which runs the user's voices and speech-to-text.
- **Services:** don't start or stop any systemd unit or docker container, except for the final restore below. The
  user's live server was stopped for this round. Its exact config is in `perf/live/` and its binaries are snapshotted
  in `~/inkling-live-bin`. Don't touch that folder.
- **Build:** `-DCMAKE_CUDA_COMPILER=$HOME/opt/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86`.
- **Profiling:** nsys and ncu are available. nsys needs `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`.
- **Commits:** commit in small steps and tag milestones `p5-<what>-<N>tok`. Never push to `ggml-org` or any upstream,
  and never open PRs or issues anywhere. Pushing to the user's fork is done separately; don't push.
- **Handoff:** keep `perf/HANDOFF.md` current after every meaningful step: done, in flight, next, and the exact commands
  and numbers. A usage limit may stop you at any moment, and a different agent will continue from it.
- **Limits:** the user's daily Claude usage matters. Work efficiently, and avoid re-reading large files or logs
  needlessly.
- **Stop signal (Claude agents):** before each build, benchmark or edit batch, check whether `perf/STOP` exists. If
  it does, finish only the command in flight, update `perf/HANDOFF.md`, commit, and end your turn with a three-line
  status. A watcher creates this file when the user's Claude usage window nears its limit, and a Codex agent then
  continues the round.

## Finishing

When the round is complete:

1. Write the final phase 5 section of RESULTS.md: tables, what worked, what didn't, sources, and risks.
2. Run `bash perf/live/restore-live.sh` to give the user their live server back. It is idempotent.
3. Report: best numbers in the tables above, the commits and tags, and what to deploy, including whether the live
   server should switch to a new build or model and with which exact command. Don't switch it yourself.
