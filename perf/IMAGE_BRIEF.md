# Image input brief: Inkling 2 Mini vision on the live release

Goal: the live endpoint (port 8031, `perf/live/deploy-p9.sh`) accepts images, without losing text speed or exactness, and
OpenCode can send them.

1. Server: add `--mmproj ~/models/inkling-2-mini/mmproj-Inkling-2-Mini-Q8_0.gguf` to the release flags (the projector type
   `PROJECTOR_TYPE_INKLING2V` is in `tools/mtmd`). Start a test server on another port with the p9 snapshot + release flags +
   mmproj, 64K context. Note VRAM.
2. Function: send real images through `/v1/chat/completions` (OpenAI `image_url` with a data URI): a screenshot with text,
   a photo-like image, a chart; check the answers describe them correctly (read text from the screenshot verbatim). Our fused
   prefill / GPU-resident embedding table were built and tested for token input only: image embeddings go through the
   embedding-input path; if anything is wrong (garbage descriptions, crashes, assertion failures), find and fix it.
3. Exactness and speed: with an image in the context, DSpark vs same-flags plain at temperature 0 must match (full message
   + token IDs) on 2–3 prompts; text-only DSpark speed with the projector loaded must be within noise of p9 (code/edit
   3,000-token cells); report image prefill time.
4. OpenCode: set the model's input modalities in `~/Code/flight-sim-reproduction/opencode.json` (and note the snippet for other
   projects) so OpenCode sends pasted/attached images (check the OpenCode config schema in `~/Code/opencode-tps` for the exact
   key, e.g. `modalities` / `attachment`), and verify with `opencode run` plus an attached image file that the model sees it.
5. Stage `~/inkling-live-bin-p9v-<commit>` (or reuse the p9 snapshot if no code changed) and `perf/live/deploy-p9v.sh`
   (p9 flags + mmproj, fallback restore-p5). Do NOT run it.
Rules: GPU1 only (GPU0 runs the user's voices); the live server is stopped by the orchestrator; don't start/stop units or
containers; never push; never open PRs/issues/comments anywhere; small commits, tags `p9v-*`; concise report.
