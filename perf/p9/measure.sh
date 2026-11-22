#!/usr/bin/env bash
# Run one measurement at a time from the repository root.
set -euo pipefail
export P8_GPU=1
source perf/p8bc/env.sh
stage=${1:?memory, prefill, decode, profile, ttft or ppl}
out=${2:?output directory}
libs=${3:?runtime library directory}
kv=${4:-q8_0}
export LD_LIBRARY_PATH="$libs:$PWD/build/bin"
mkdir -p "$out" build/p9
args=(-m "$Q2" -ngl 99 -ctk "$kv" -ctv "$kv")
case $stage in
    prefill)
        build/bin/llama-bench "${args[@]}" -fa 1 -p 4096,16384,32768 -n 0 -b 2048 -ub 512 -r 3 -o json >"$out/$kv.json" 2>"$out/$kv.log" ;;
    decode)
        env -u IK_MR -u IK_MMA -u IK_QUEUE_INPUTS -u IK_WINDOW_COLS -u GGML_CUDA_GRAPH_CHUNK_SIZE build/bin/llama-bench "${args[@]}" -fa 1 -p 0 -n 128 -d 0,4096,16384,32768 -r 3 -o json >"$out/decode.json" 2>"$out/decode.log" ;;
    profile)
        profile="$PWD/build/p9/$(basename "$out")-$kv"
        nsys="$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys"
        "$nsys" profile -t cuda --cuda-graph-trace=node -o "$profile" --force-overwrite true build/bin/llama-bench "${args[@]}" -fa 1 -p 4096 -n 0 --no-warmup -b 2048 -ub 512 -r 1 >"$profile.log" 2>&1
        "$nsys" stats --force-export=true --report cuda_gpu_kern_sum --force-overwrite=true --format csv -o "$profile" "$profile.nsys-rep" >>"$profile.log" 2>&1
        cp "${profile}_cuda_gpu_kern_sum.csv" "$out/profile-$kv.csv" ;;
    memory)
        for ctx in 16384 32768 65536; do
            log="build/p9/$(basename "$out")-$kv-$ctx.log"
            "${SPEC_SERVER_BIN:-build/bin/llama-server}" "${args[@]}" -fa on -c "$ctx" -np 1 -b 2048 -ub 512 -lv 5 --backend-sampling --host 127.0.0.1 --port 18439 >"$log" 2>&1 &
            pid=$!
            trap 'kill "$pid" 2>/dev/null || true; wait "$pid" || true' EXIT
            for i in {1..120}; do
                if curl -fsS --max-time 1 http://127.0.0.1:18439/health >/dev/null 2>&1; then break; fi
                kill -0 "$pid"
                sleep 0.5
            done
            curl -fsS http://127.0.0.1:18439/health >/dev/null
            {
                rg 'compute buffer size =|KV buffer size|n_ctx_slot =' "$log"
                nvidia-smi -i 1 --query-gpu=uuid,memory.used,memory.free --format=csv
                nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader | awk -v p="$pid" '$1+0==p {print; found=1} END {exit !found}'
            } >"$out/mem-$kv-$ctx.txt"
            rg -q 'CUDA0 compute buffer size =' "$log"
            kill "$pid"; wait "$pid" || true; trap - EXIT
        done ;;
    ttft)
        for width in 0 6; do
            TTFT_SIZES=16384 DRAFT_WIDTH=$width P8_LIBS="$libs" bash perf/p8bc/ttft.sh "p9-$(basename "$out")-$width" >"$out/ttft-$width.txt"
        done ;;
    ppl)
        build/bin/llama-perplexity "${args[@]}" -fa on -c 2048 -b 2048 -ub 512 --chunks 8 -f perf/ppl.txt >"$out/ppl.log" 2>&1
        rg 'Final estimate' "$out/ppl.log" ;;
    *) exit 2 ;;
esac
