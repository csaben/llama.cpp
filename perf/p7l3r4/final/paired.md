# Temperature-matched release crossovers

All sixteen processes started at 63 C on GPU0. A disables Round 4 with `IK_GLUE_FAST=0`. B enables it. Rates are tok/s, with three tg128 repetitions per process.

| Comparison | A1 | B1 | B2 | A2 | A mean | B mean | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| [paired-0](paired-0.txt) | 224.42 | 233.05 | 233.49 | 224.27 | 224.34 | 233.27 | +3.98% |
| [paired-4096](paired-4096.txt) | 207.17 | 214.11 | 213.90 | 207.87 | 207.52 | 214.00 | +3.12% |
| [paired-16384](paired-16384.txt) | 194.64 | 200.26 | 198.78 | 194.29 | 194.46 | 199.52 | +2.60% |
| [paired-32768](paired-32768.txt) | 179.25 | 184.94 | 184.56 | 179.62 | 179.44 | 184.75 | +2.96% |
