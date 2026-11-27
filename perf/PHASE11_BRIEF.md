# Phase 11 brief: two GPUs over P2P, and tensor cores at full rate (overnight 2026-11-27)

Base: the live p9 release (source `72fd097`, snapshot `~/inkling-live-bin-p9-72fd097`, flags in
`perf/live/deploy-p9v-uncensored-128k.sh`, 128K context, alias `inkling-mini`). Read `perf/HANDOFF.md`, `perf/RESULTS.md`
(Phases 7-9) and `perf/PHASE10_NUMBERS.md` first. Primary target: the uncensored 2-bit model
`~/models/inkling-2-mini/Inkling-2-Mini-AtomicChat-Uncensored-Q2K-Q2_0-fit.gguf` (identical output head to the Q2K-Q2_0-fit
target, so the folded 32K drafter is valid). Report the IQ4_NL-dense 4-bit too if time allows.

## What changed on this machine since Phase 10
1. **P2P between the two 3090s.** Driver 615.71.09 with aikitoria's P2P-patched open kernel modules
   (github.com/aikitoria/open-gpu-kernel-modules, branch 615.71.09-p2p; rebuilt for kernel 7.0.0-44), IOMMU passthrough.
   `cudaDeviceCanAccessPeer` is 1 both ways. Measured with `perf/p2p/p2p-check.cu` (binary `~/deploy/p2p/p2p-check`):
   9 KB round trip 26 -> 13.5 us, in-kernel peer write (one GPU's kernel stores straight into the other's memory)
   ~2.9 us, 256 MB copy 5.8 -> 13.2 GB/s. Before this, the cards could not talk directly at all.
2. **gpu-unlock** (github.com/zagitta/gpu-unlock @7eb51bc, v2.0.0) sets the GA102 tensor-core speed selectors to full
   rate at boot. 8192^2 GEMM (`~/deploy/gpu-unlock-bench.py`): BF16 GPU0 73.9 -> 119.5, GPU1 77.7 -> 131.2 TFLOPS;
   FP16 similar. Bandwidth is unchanged, so plain 1-GPU decode should barely move; prefill, DSpark verify (2-7 columns)
   and anything MMA-bound may.
3. Both GPUs are free all night (the orchestrator paused the voice/ASR/UVR units and the live server).

## Stream A: two-GPU decode (worktree `~/llama.cpp-fast`, branch `inkling-3090-p11a`, tags `p11a-*`)
Goal: plain decode well above **241 tok/s** and DSpark above **351** (short edit, cap per policy) on the 2-bit target,
without losing 128K context. Decode is bandwidth bound (~62% of one card's ceiling on 2-bit), so the lever is reading
half the weights per card and paying only microseconds per exchange. Candidates, measure before building big:
- **Tensor parallel** for the expert/dense matvecs and attention heads: split rows (or experts) across the cards, each
  card reads half; exchange partial vectors with in-kernel peer writes plus a flag (no cudaMemcpyPeer, no NCCL launch
  overhead), ideally inside the existing CUDA graphs. Count exchanges per token x ~3 us against the bandwidth saved;
  report that budget first.
- **Drafter on the other GPU**: the DSpark drafter on GPU0, the target on GPU1, overlapping draft and verify; ship the
  feature rows / draft IDs over P2P.
- Layer/pipeline split alone does not speed up batch-1 decode; use it only if it frees VRAM for something that does
  (e.g. longer context or a bigger verify width).
Exactness: a pure placement change (drafter on another card, whole-layer split) must be bit-identical to the 1-GPU p9
build. If tensor parallel changes FP32 reduction order, it must still be exact DSpark-vs-plain on the same build (the
release matrix), PPL within 0.1% of p9 (prefill 3.0530, decode 4.1979 / 4.2038), and you report its divergence from
the 1-GPU outputs. Keep the 1-GPU path untouched behind the default flags.

## Stream B: full-rate tensor cores (worktree `~/llama.cpp-p11b`, branch `inkling-3090-p11b`, tags `p11b-*`)
Re-measure p9 under gpu-unlock first (plain tg at 0/4K/16K/32K/64K, prefill 4K/16K/64K, DSpark short matrix, nsys
prefill and verify profiles), then retune what is now MMA-bound: prefill GEMM tile shapes / Stream-K partition, the
attention scan, the DSpark verify kernels at widths 2-7, and the DSpark policy's cost model (verify got cheaper, so the
best cap may rise). Exact or PPL-gated as above.

## GPU sharing (both streams run at once)
- GPU1 belongs to Stream A. Stream B uses GPU0 only (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`).
- Any process of Stream A that touches GPU0, and every Stream B GPU0 process, runs under
  `flock /home/arelius/llama.cpp-fast/perf/p11/gpu0.lock <cmd>`. Hold it for one bench/check at a time (minutes, not
  hours), never while idle or reading code. Don't kill processes you did not start.
- Use ports 18500-18549 (A) and 18550-18599 (B). Port 8031/8032 are the live server's; don't touch them.

## Verification (both streams; the orchestrator re-checks before anything goes live)
Use the p9 tools: `perf/p9/matrix/verify.sh` (release matrix: drafted replies vs same-flags plain, every token ID),
the rollback/boundary battery and `mr_check`, `test-backend-ops` for changed ops, PPL. A result counts only with the
paired old/new/new/old benchmark and matching outputs. Save evidence under `perf/p11/a/` or `perf/p11/b/`.

## Deadline and deliverables
- 07:30 EST: stop optimizing. Stage the best verified build as `~/inkling-live-bin-p11<a|b>-<commit>` + sha256 manifest
  + `perf/live/deploy-p11<a|b>.sh` (128K, port 8031, alias inkling-mini, fallback
  `perf/live/deploy-p9v-uncensored-128k.sh`; a 2-GPU build may use both cards). Don't run it.
- By 07:45: `perf/p11/<a|b>/REPORT.md` with a before/after table (plain decode by depth, DSpark short/long, prefill,
  PPL, max context, VRAM per card), what failed and why, and sources (aikitoria p2p, zagitta/gpu-unlock, anything else
  you used). Update HANDOFF.md ("Phase 11") and RESULTS.md. Commit.
- If `perf/STOP` exists, finish the current step, write the report, commit and exit.

Rules: never push; never open PRs/issues/comments on ggml-org, unslothai or any upstream; don't start/stop systemd units
or containers; small commits; cite sources; concise final report.
