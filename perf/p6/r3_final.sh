#!/bin/bash
# Round 3 checks: window equality, plain decode at depth, PPL, backend ops. usage: perf/p6/r3_final.sh  (outputs perf/p6/r3_*.txt)
cd "$(dirname "$0")/../.." && . perf/p5env.sh
export LD_LIBRARY_PATH=$PWD/build/bin
B="-m $Q2 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0"
{ for a in "300 24" "300 16" "1500 12"; do IK_MMA=3 IK_MR=1 build/bin/mr_check $Q2 perf/ppl.txt $a 2>&1 | grep -E 'ms per step|rows in'; done; } > perf/p6/r3_mr_check.txt
{ for e in "IK_NOP=1" "IK_MMA=3 IK_MR=1"; do echo "== $e"; env $e build/bin/llama-bench $B -p 0 -n 128 -d 0,4096,16384,32768 -r 3 2>/dev/null | grep tg128; done; } > perf/p6/r3_decode.txt
{ for e in "IK_NOP=1" "IK_MMA=3 IK_MR=1"; do
    echo "== $e prefill 8x2048"; env $e build/bin/llama-perplexity -m $Q2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 2048 -b 2048 --chunks 8 2>&1 | grep 'Final estimate'
    echo "== $e decode 2x1024 -ub 1"; env $e build/bin/llama-perplexity -m $Q2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -f perf/ppl.txt -c 1024 -b 1024 -ub 1 --chunks 2 2>&1 | grep 'Final estimate'
  done; } > perf/p6/r3_ppl.txt
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_ID,INKLING_ROUTER,INKLING_ATTN 2>&1 | grep -E 'tests passed|Backend CUDA0: |FAIL' | tail -5 > perf/p6/r3_backend_ops.txt
