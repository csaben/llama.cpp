# Phase 8 brief, streams B and C (GPU0): better drafts, faster prefill

Work in the worktree `~/llama.cpp-p8bc` (branch `inkling-3090-p8bc`, from `964319a` = the p7c release evidence on top of
source `401460d`). Stream A (glue-kernel latency) runs at the same time in `~/llama.cpp-fast` on GPU1: never edit that
tree, never use GPU1. Read the Phase 6–7c sections of `perf/RESULTS.md` and `perf/HANDOFF.md` first. Keep notes in
`perf/HANDOFF_P8BC.md` and results in a "Phase 8 B/C" section of `perf/RESULTS_P8BC.md`.

## Stream B: more accepted drafts on new code, prose and agent turns (exact)
Release numbers (tok/s, plain vs DSpark): short code 212 -> 252, short prose 211 -> 220; long code 201 -> 277. Edits
are at 343 / 306 because copied text drafts well. Ideas, measure and rank:
1. **Prompt-lookup / n-gram drafting** from the context (Saxena, "Prompt Lookup Decoding", github.com/apoorvumang/
   prompt-lookup-decoding; llama.cpp's n-gram / lookup drafting) combined with DSpark: per round, choose whichever
   drafter's candidates are expected to pay (basalt chooses between DSpark and prompt lookup: +13% on a copy prompt,
   `src/spec/draft_policy.cpp`, MIT). Agent turns (tool results, file contents, diffs echoed back) are where this pays.
2. Longer windows only where acceptance is high (e.g. n=4–6 on runs of matching lookups), policy-guarded.
3. Anything else the per-round profile shows for code/prose acceptance.
Measure with the 7c standard plus one **agent-style replay**: a multi-turn OpenCode-like request with tool results
(build it from `perf/longctx.py` or the 64K tool-history replay used in round 4). Output must stay byte-identical to
same-flags plain on every prompt (full message + token IDs, 3,000 and 384 tokens).

## Stream C: prefill / time to first token (after B, or interleaved if B is blocked)
A 16K-token agent prompt takes several seconds. Profile prefill at 4K / 16K / 32K (`llama-bench -p 4096,16384 -n 0`,
nsys) and attack the top items: the expert GEMMs (ingine `prefill/wgemm.cu`: experts decoded straight into tensor-core
fragments, 32-row tiles, no predicated work: +20% there; Apache-2.0, keep its notice if adapting), dense GEMMs,
attention prefill. Exact or PPL-gated (prefill PPL 3.0530 must hold within 0.5%; report).

## Rules
- **GPU0 only** (`export CUDA_VISIBLE_DEVICES=0` after `. perf/p5env.sh`). The user's GPU0 services are paused for this.
  Measure your own GPU0 baselines first.
- Build here only (`cmake -B build ...` with `~/llama.cpp-fast/build/CMakeCache.txt` flags).
- Never push; never open PRs/issues/comments anywhere. Small commits, tags `p8b-*` / `p8c-*`. Cite every source.
- Don't start/stop systemd units or containers. When done, leave a clean tree and a concise report (no deploy script;
  the orchestrator merges with stream A).
