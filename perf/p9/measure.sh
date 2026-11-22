#!/usr/bin/env bash
# Run one measurement at a time from the repository root.
set -euo pipefail
export P8_GPU=1
source perf/p8bc/env.sh
stage=${1:?memory, prefill, decode, profile, ttft, near-full or ppl}
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
        tokens=${P9_PROFILE_TOKENS:-4096}
        profile="$PWD/build/p9/$(basename "$out")-$kv-$tokens"
        nsys="$HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys"
        "$nsys" profile -t cuda --cuda-graph-trace=node -o "$profile" --force-overwrite true build/bin/llama-bench "${args[@]}" -fa 1 -p "$tokens" -n 0 --no-warmup -b 2048 -ub 512 -r 1 >"$profile.log" 2>&1
        if [[ ! -s $profile.nsys-rep ]]; then
            env -u LD_PRELOAD "$(dirname "$nsys")/../host-linux-x64/QdstrmImporter" -i "$profile.qdstrm" -o "$profile.nsys-rep" >>"$profile.log" 2>&1
        fi
        env -u LD_PRELOAD "$nsys" stats --force-export=true --report cuda_gpu_kern_sum --force-overwrite=true --format csv -o "$profile" "$profile.nsys-rep" >>"$profile.log" 2>&1
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
    near-full)
        log="build/p9/$(basename "$out")-64k-memory.csv"
        nvidia-smi -i 1 --query-gpu=timestamp,pci.bus_id,memory.used,memory.free --format=csv,nounits -lms 250 >"$log" &
        monitor=$!
        trap 'kill "$monitor" 2>/dev/null || :; wait "$monitor" 2>/dev/null || :' EXIT
        TTFT_SIZES=64512 DRAFT_WIDTH=6 P8_LIBS="$libs" bash perf/p8bc/ttft.sh "p9-$(basename "$out")-64k" >"$out/ttft-64k.txt"
        kill "$monitor"; wait "$monitor" || :; trap - EXIT
        awk -F', ' 'NR == 2 {min = $4} NR > 1 {if ($4 < min) min = $4; if ($3 > peak) peak = $3} END {if (NR < 2) exit 1; print "peak used MiB", peak; print "minimum free MiB", min}' "$log" >"$out/memory-64k.txt" ;;
    ppl)
        build/bin/llama-perplexity "${args[@]}" -fa on -c 2048 -b 2048 -ub 512 --chunks 8 -f perf/ppl.txt >"$out/ppl.log" 2>&1
        rg 'Final estimate' "$out/ppl.log" ;;
    *) exit 2 ;;
esac
