# Inkling 2 Mini image input on GPU1

The 64K release passed on port 18491 with the Q8_0 projector and the p9 snapshot at `72fd097e`. All nine snapshot files matched `perf/p9/snapshot.sha256`. Total GPU1 usage was 1,254 MiB before startup, 16,177 MiB at readiness, and 16,371 MiB after the first three image requests.

Each image was sent as a data URI to `/v1/chat/completions` with temperature 0, seed 1 and prompt caching disabled. DSpark and same-flags plain matched on every full assistant message, including reasoning, and all 416 output token IDs. Each completion stopped naturally.

| Image | Verified answer | Image tokens | Image embedding decode | Whole prompt prefill | Output IDs |
|---|---|---:|---:|---:|---:|
| 660 x 320 browser screenshot | `Quartz Harbor 47` verbatim | 136 | 78 ms | 274 ms | 79 |
| 550 x 368 landscape photo | A blue fjord between steep green mountains, with a rocky outcrop in front | 140 | 70 ms | 190 ms | 180 |
| 660 x 390 bar chart | North, 61 | 170 | 82 ms | 198 ms | 77 |
| 1536 x 864 browser screenshot | `Copper Ridge 82` verbatim | 858 | 323 ms | 551 ms | 80 |

Times are from the DSpark timing run. The screenshot was the first request. The desktop screenshot exceeded the 512-token microbatch and exercised the existing embedding-input path across microbatches.

Text-only DSpark used the code and edit prompts from `perf/spec_probe.py`, three uncached greedy completions of 3,000 tokens per cell. Median code / edit throughput was **277.6 / 311.3 tok/s** with the projector and **280.7 / 313.0 tok/s** on the first p9 control. A second p9 control measured **278.5 / 310.8 tok/s**. With the projector, both cells were within 0.4% of that control and the published 64K p9 results. Text messages and output token IDs also matched across configurations.

OpenCode read `Quartz Harbor 47` from the attached PNG with tools denied. The flight-sim project's saved model URL is port 8031. Only the test invocation overrode it to port 18491. Other projects can copy the `attachment` and `modalities` fields from [the model configuration](opencode-inkling.json).

[The staged launcher](live/deploy-p9v.sh) uses the existing p9 runtime and the phase 5 fallback. Response JSON, image fixtures, server logs and text benchmark outputs are in `build/p9v/` on the test host.
