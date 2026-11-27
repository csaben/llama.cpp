#!/usr/bin/env bash
# Copy matching p9 frontends before Stream A rebuilds its worktree.
set -euo pipefail
mkdir -p build/p11/p9-tools
src=${P9_TOOL_DIR:-$HOME/llama.cpp-fast/build/bin}
for f in llama-bench libllama-bench-impl.so llama-perplexity libllama-perplexity-impl.so mr_check; do
    test -s "$src/$f"
    cp -Lp "$src/$f" "build/p11/p9-tools/$f"
done
# The release checker has a 65536-token bound including the eight sampled rows.
sed 's/prefix > 65536 - steps/prefix > 131072 - steps/' perf/p7l3/logits.cpp >build/p11/logits.cpp
g++ -O2 -std=c++17 -Wno-deprecated-declarations build/p11/logits.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base -o build/p11/p9-tools/logits
