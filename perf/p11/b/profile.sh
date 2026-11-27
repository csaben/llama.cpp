#!/usr/bin/env bash
set -eu
nsys="$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys"
dir=build/p11/profiles
mkdir -p "$dir"
kind=${1:?prefill or verify}
tag=${2:-p9}
base="$dir/$tag-$kind"
case $kind in
    prefill) args=(bench -p 4096 -n 0 -r 1 --no-warmup) ;;
    verify) args=(probe "$dir" "$tag-verify" 6 edit 1 160) ;;
    *) exit 2 ;;
esac
perf/p11/b/gpu.sh "$nsys" profile --sample=none --cpuctxsw=none -t cuda --cuda-graph-trace=node --force-overwrite=true -o "$base" perf/p11/b/run.sh "${args[@]}" >"$base.log" 2>&1 || test -s "$base.qdstrm"
if [[ ! -s $base.nsys-rep ]]; then
    env -u LD_PRELOAD "$(dirname "$nsys")/../host-linux-x64/QdstrmImporter" -i "$base.qdstrm" -o "$base.nsys-rep" >>"$base.log" 2>&1
fi
env -u LD_PRELOAD "$nsys" stats --force-export=true --report cuda_gpu_kern_sum --force-overwrite=true --format csv -o "$base" "$base.nsys-rep" >>"$base.log" 2>&1
test -s "${base}_cuda_gpu_kern_sum.csv"
cp "${base}_cuda_gpu_kern_sum.csv" "perf/p11/b/$tag-$kind-kernels.csv"
