# Phase 10 brief: publishable numbers (2-bit and 4-bit; short and big context)

Measure, don't optimize. The user will post these, so every number must be defensible: say exactly what was measured.
Build: the live release `p9` (source `72fd097`, snapshot `~/inkling-live-bin-p9-72fd097`, flags in `perf/live/deploy-p9.sh`).
Models (`~/models/inkling-2-mini/`):
- **2-bit**: `Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf` (9.94 GB; prefill PPL 3.0530, +3.8%).
- **4-bit**: `Inkling-2-Mini-IQ4_NL-dense-iq4nl.gguf` (17.07 GB; IQ4_NL experts and dense; PPL +0.2% per RESULTS.md).
  Also, if time allows, the near-exact `Inkling-2-Mini-IQ4_NL-dense-q6k.gguf` (Q6_K dense).
Drafter: `dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf` (folded against the 2-bit target's head: check whether it is
valid for the 4-bit targets; if the fold is target-specific, build the matching fold with `perf/p6/dspark_fold.py` or use
the plain `dspark-Inkling-2-Mini-Q8_0.gguf`, and say which).

## Measure, for each model
1. **Max context that fits on one 24 GB 3090** with the release settings (Q8_0 KV, drafter loaded): try 262144, then
   step down (196608, 131072, 98304). Report VRAM used and free.
2. **Plain decode** `llama-bench -p 0 -n 128 -r 3` at depths 0, 4K, 16K, 32K, 64K, 128K and the max (as far as fits),
   release env, plus **prefill** `-p 4096,16384` and prefill throughput when filling the max context.
3. **Served DSpark** (release flags, temperature 0, 3,000-token replies; the `perf/p6/r4_spec.sh` / p9 matrix
   harness): code / edit / prose / agent replay at short context, and an edit at 64K, 128K and max context if they fit.
   Also one temperature 1.0 run per model (code + edit, 1,200 tokens, seeds 11 and 22) for the realistic number.
4. **Correctness at big context**: our fused attention was only validated to ~50K. At 128K and max: a needle-in-a-
   haystack retrieval test (several needles at different depths, plain and DSpark), `mr_check` at large prefixes if
   feasible, and DSpark vs same-flags plain equality (full message + token IDs) on the long-context prompts. For the
   4-bit model, also run the window exactness check (`mr_check`) and the 3,000-token equality matrix, since the tensor-
   core verify kernels were built for the 2-bit experts. If anything is wrong, stop and report it rather than tuning.
5. Note the GPU power limit in use (`nvidia-smi -q -d POWER`); don't change it.

## Output
A short `perf/PHASE10_NUMBERS.md` with one table per model (context depth vs plain decode / DSpark / prefill), the max
context and VRAM, the correctness results, and the exact commands; commit it (tag `p10-numbers`). Rules: GPU1 only
(GPU0 runs the user's voices); the live server is stopped for you; don't start/stop units or containers; never push;
never open PRs/issues/comments anywhere. Concise report.
