# A/B brief: does the near-exact model build a better flight sim than the 2-bit model?

Goal: compare output quality, not speed. Same harness, same prompt, fresh workspace each run.

- Prompt (exactly): `I'm on a plane with no Wi-Fi and bored. Can you build me a simple flight sim?`
- Harness: OpenCode CLI, headless: `cd ~/Code/flight-sim-reproduction && opencode run --format json -m localinkling/inkling-mini "<prompt>"`
  (the workspace's `opencode.json` points at http://127.0.0.1:8031/v1). Before every run: `inkling-demo-reset` (archives
  the previous attempt and restores AGENTS.md, opencode.json, node_modules/three). After every run, move the produced
  files to `~/Code/.flight-sim-attempts/ab/<model>-<n>/` (plus a symlink `node_modules -> ~/Code/flight-sim-reproduction/node_modules`
  so it runs standalone), and save the JSON event log, wall time, number of tool calls, and the server's per-request
  tok/s from `journalctl` there too.
- Server on port 8031 (the live unit `inkling-3090-experiment` is stopped by the orchestrator; run your own server with
  the release flags from `perf/live/deploy-p9.sh`, same port, same sampling defaults, i.e. send no sampling overrides):
  - **A: 2-bit** `Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf` with drafter `dspark-Inkling-2-Mini-Q8_0-fold-short32k.gguf`.
  - **B: near-exact** `Inkling-2-Mini-IQ4_NL-dense-q6k.gguf` with drafter `build/phase10/dspark-q6-short32k.gguf`
    (the fold phase 10 made for it), context as large as fits with the release settings (≥ 64K if possible; else the
    largest that fits, say which).
- 3 runs per model, alternating A, B, A, B, A, B. If a run stalls (no progress for 10 minutes) or exceeds 25 minutes, stop
  it and record that as an outcome.
- For each attempt, an automated check (Playwright with the local chromium headless shell at
  `~/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell --no-sandbox`,
  serve the attempt folder with `python3 -m http.server` on localhost): does the page load without console errors, is
  there a rendered 3D scene (non-blank canvas), does holding throttle/pitch keys for a few seconds change the view (the
  plane moves), and does it survive 30 s of play without errors. Save screenshots at 0 s, 5 s and 20 s of flight.
- Write `~/Code/.flight-sim-attempts/ab/SUMMARY.md`: a table per attempt (model, wall time, tool calls, files, lines of
  code, loads?, flies?, errors, screenshot paths) and one paragraph of honest observations. Do not judge "feel" beyond
  what the checks show; the user will fly them.

Rules: GPU1 only (GPU0 runs the user's voices). Don't start/stop systemd units or containers (your server is a plain
process you start and stop yourself). Don't modify the reference at `~/Code/.flight-sim-attempts/tm-reference`. Never push or
open PRs/issues anywhere. When finished, stop your server so port 8031 is free. Concise report.
