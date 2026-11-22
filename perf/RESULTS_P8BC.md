# Phase 8 B/C

## Controls

Physical GPU 0 (RTX 3090, 01:00.0), release Q2_K/Q2_0 target, folded 32K DSpark shortlist, Q8_0 KV, 64K slot, backend greedy sampling, cache off, EOS suppressed. Short rates are medians of three uncached requests unless indicated. Full sorted message JSON and little-endian int32 emitted IDs are checked on every repeat. The agent replay includes an assistant read call, a numbered tool result and a user follow-up asking for the edited file. [Runner](p8bc/probe.sh), [flags](p8bc/env.sh), [request and capture](spec_probe.py), [comparator](p6/r4_cmp.py).

Frozen baseline uses the Phase 7c server and all its matching libraries from `~/inkling-live-bin-p7c-401460d`. It already proposes history after a 12 or 16 token match. Historical GPU 1 rates are in the [Phase 7c matrix](RESULTS.md#phase-7c). The local build uses the SM86 CUDA 12.8 Release flags documented in [RESULTS.md](RESULTS.md).

## Sources

- Prompt lookup copies candidates from an earlier matching token sequence. [Saxena, Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding), [llama.cpp n-gram implementation](../common/ngram-map.cpp).
- Basalt compares estimated committed tokens per unit time across sources and tracks conditional success by match length. [MIT draft policy](https://github.com/jesdga95/basalt/blob/ae82a4841c95f6a05ca8ca2331ef6af7ca844aef/src/spec/draft_policy.cpp).
- Ingine groups routed rows and decodes its packed weights into BF16 MMA fragments. Its data layout and precision differ from this branch's GGUF/Q8 activation MMQ. [Apache-2.0 expert kernel](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/306eaa8a90c9840f042a59d565f6dc4536bbb5ea/native/src/kernels/prefill/wgemm.cu).

## Frozen GPU 0 baseline

Decode tok/s, 384 / 3,000 emitted tokens. All six drafted messages and ID streams matched plain at both lengths. The five standard plain prompts also reproduced the Phase 7c references across GPUs.

| Prompt | Plain | Release cap 3 |
|---|---:|---:|
| Code | 208.5 / 191.7 | 241.9 / 262.6 |
| Edit | 198.7 / 191.1 | 327.7 / 293.1 |
| Prose | 208.1 / 196.5 | 209.1 / 283.6 |
| Edit at 16K | 180.3 / 178.0 | 278.8 / 284.9 |
| Edit at 32K | 166.2 / 164.9 | 243.5 / 286.7 |
| Agent | 191.8 / 192.1 | 280.0 / 308.8 |

[384-token captures](p8bc/baseline/384/), [3,000-token captures](p8bc/baseline/3000/), [GPU and driver](p8bc/baseline/device.csv).
