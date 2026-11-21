#!/usr/bin/env bash
# Reproduce the dense-IQ4_NL candidate from the published IQ4_NL mixture.
set -eu
if [[ $# != 2 || ! -f $1 || -e $2 ]]; then
    echo "usage: bash perf/quantize-inkling-dense.sh ORIGINAL.gguf NEW.gguf" >&2
    exit 2
fi
exec build/bin/llama-quantize --allow-requantize \
    --output-tensor-type iq4_nl --token-embedding-type q8_0 \
    --tensor-type 'attn_(q|k|v|output|gate|rel|sink_proj|k_conv_proj|v_conv_proj)\.weight=iq4_nl' \
    --tensor-type 'blk\.(1[0-9]|[0-9])\.ffn_.*_exps=iq4_nl' \
    --tensor-type 'blk\.2[0-3]\.ffn_.*_exps=q8_0' "$1" "$2" iq4_nl 8
