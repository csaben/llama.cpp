#!/bin/bash
# usage: perf/p6/prof_check.sh <name> <sizes> <n_windows> [rows]   (. perf/p5env.sh first) -> per-window-step kernel table of mr_check
O=${PROF_DIR:-/tmp/claude-1000/-home-arelius-deploy/453a4d12-6978-485f-8868-dd0a2e664559/scratchpad/p6prof}; mkdir -p $O
R=$(cd "$(dirname "$0")/../.." && pwd); N=$1; rm -f $O/$N.*
MR_ROLLBACK=0 MR_SIZES=$2 IK_MR=${IK_MR-1} LD_LIBRARY_PATH=$R/build/bin $HOME/opt/cuda-12.8/nsight-systems-2024.6.2/bin/nsys profile -t cuda --cuda-graph-trace=node -o $O/$N --force-overwrite true --export sqlite $R/build/bin/mr_check $Q2 $R/perf/ppl.txt 300 $3 2>&1 | grep 'ms per'
python3 - $O/$N.sqlite $3 ${4:-14} <<'PY'
import sqlite3,collections,sys
c=sqlite3.connect(sys.argv[1]); NW=int(sys.argv[2])
h=collections.Counter(); t=collections.Counter()
for name,us,gx,gy,gz in c.execute("select s.value,(k.end-k.start)/1000.0,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id"):
    key=(name[:34],gx,gy,gz) if name.startswith('ik_mmvq_mr') or name.startswith('mul_mat_vec_q') else (name[:34],)
    h[key]+=1; t[key]+=us
for k,v in sorted(t.items(), key=lambda x:-x[1])[:int(sys.argv[3])]: print(f"{v/NW/1000:7.3f} ms/window {h[k]/NW:7.1f}/window {v/h[k]:7.1f} us", k)
PY
