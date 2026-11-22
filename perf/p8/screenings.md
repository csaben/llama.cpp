# Phase 8 isolated kernel trials

Each row switches runtime libraries in a frozen/candidate/candidate/frozen crossover on physical GPU1. Rates are the mean of the two process means (three tg128 repetitions each). Local tags preserve the trial source. The release uses the controller stripe pack and the long-depth portion of the gate permutation. The MR-specific trials are documented separately in RESULTS.md.

| Trial | Change | Depth | Frozen tok/s | Trial tok/s | Change |
|---|---|---:|---:|---:|---:|
| [attention](attention/) | More gate blocks and sliding attention warps | 0 | 241.39 | 240.82 | -0.24% |
| [attention](attention/) | More gate blocks and sliding attention warps | 16384 | 205.84 | 200.93 | -2.39% |
| [owner](owner/) | Skip Q8 work on non-owner router warps | 0 | 242.49 | 242.50 | +0.00% |
| [route2](route2/) | Two expert rows per router warp | 0 | 241.55 | 239.74 | -0.75% |
| [rank1](rank1/) | One controller rank per block | 0 | 240.57 | 239.30 | -0.53% |
| [wordload](wordload/) | Fetch adjacent IQ4 blocks with aligned word loads | 0 | 241.09 | 239.51 | -0.65% |
| [wordload](wordload/) | Fetch adjacent IQ4 blocks with aligned word loads | 16384 | 206.03 | 204.44 | -0.77% |
| [warpmerge](warpmerge/) | Parallel head-local split coefficients | 0 | 240.59 | 240.52 | -0.03% |
| [warpmerge](warpmerge/) | Parallel head-local split coefficients | 16384 | 206.05 | 206.32 | +0.13% |
| [current](current/) | Skip current-cell normalization on empty splits | 16384 | 206.24 | 206.38 | +0.07% |
| [redux](redux/) | Use hardware warp reduction for top eight | 0 | 241.29 | 240.99 | -0.13% |
| [assist24](assist24/) | Move 24 gate rows to spare scan warps | 0 | 243.20 | 243.31 | +0.04% |
| [assist48](assist48/) | Move 48 gate rows to spare scan warps | 0 | 242.09 | 228.63 | -5.56% |
| [pack16](pack16/) | Pack controller stripes on a 16-tile grid | 0 | 241.40 | 242.16 | +0.32% |
| [vpack](vpack/) | Permute packed V in registers | 0 | 240.27 | 240.18 | -0.04% |
| [vpack](vpack/) | Permute packed V in registers | 16384 | 205.43 | 205.23 | -0.10% |
| [first](first/) | Skip rescaling the first attention tile | 0 | 240.37 | 240.07 | -0.12% |
| [sidekv](sidekv/) | Move current K/V convolution to spare scan warps | 0 | 240.44 | 240.71 | +0.11% |
| [shortempty](shortempty/) | Skip work on masked short attention tiles | 0 | 241.49 | 241.22 | -0.11% |
| [masktiles](masktiles/) | Precompute sliding-window tile masks | 16384 | 206.94 | 205.63 | -0.63% |
| [full1](full1/) | One gate warp for deep full attention | 16384 | 205.97 | 201.25 | -2.29% |
| [full2](full2/) | Two gate warps for deep full attention | 16384 | 205.32 | 206.21 | +0.43% |
| [mixwarp](mixwarp/) | Replicate residual mix factors in warp registers | 0 | 240.84 | 239.12 | -0.71% |
| [routerwarp](routerwarp/) | Replicate router RMS and probability factors in warp registers | 0 | 240.25 | 240.98 | +0.30% |
| [routervote](routervote/) | Combine router probability publication with the NaN vote | 0 | 240.75 | 241.14 | +0.16% |
| [input2](input2/) | Reuse Q8 registers over two scan input rows per warp | 0 | 240.69 | 238.09 | -1.08% |
| [input2](input2/) | Reuse Q8 registers over two scan input rows per warp | 16384 | 206.44 | 203.75 | -1.30% |
| [input4](input4/) | Reuse Q8 registers over four scan input rows per warp | 0 | 239.67 | 236.04 | -1.51% |
| [input4](input4/) | Reuse Q8 registers over four scan input rows per warp | 16384 | 184.94 | 203.28 | +9.91% |
| [rankreuse](rankreuse/) | Share controller activation registers between two rank dots | 0 | 239.73 | 240.06 | +0.14% |
| [ranknorm](ranknorm/) | Share controller norms across paired rank dots | 0 | 240.48 | 240.44 | -0.01% |
| [gatecodes](gatecodes/) | Permute Q8 once per gate warp at both depths | 0 | 240.08 | 240.28 | +0.08% |
| [gatecodes](gatecodes/) | Permute Q8 once per gate warp at both depths | 16384 | 205.45 | 207.06 | +0.78% |

The last 16K control of input4 was an outlier (164.35 tok/s). Both candidate passes were also slower than the opening 205.53 control. The two-warp deep-full and replicated-router trials showed small crossover gains, with overlapping or drifting controls. Those changes were left out of the release. Trial names map to local tags `p8a-trial-<name>`.
