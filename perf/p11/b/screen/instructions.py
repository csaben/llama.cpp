from pathlib import Path
import collections,csv,re,sqlite3,subprocess
root=Path('build/p11/profiles')
out=Path('perf/p11/b/screen')
configs={}
for filename, pat in [('p9-verify', 'void ik_q2k_moe<(int)8>%'),('p9-verify','void ik_mma_moe<(ggml_type)42, (int)8>%'),('p9-prefill', 'void mul_mat_q<(ggml_type)42,%'),('p9-prefill','void mul_mat_q<(ggml_type)10,%'),('p9-verify','void inkling_attn_window_cols<(int)0>%'),('p9-prefill','%flash_attn_ext%')]:
    db=sqlite3.connect(f'file:{root/filename}.sqlite?mode=ro',uri=True)
    for label,mangled,regs,stat,dyn in db.execute('select distinct d.value,m.value,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds d on d.id=k.demangledName join StringIds m on m.id=k.mangledName where d.value like ?',(pat,)):
        configs[mangled]=(label.split('>(')[0]+'>',regs,stat,dyn)
assert len(configs)>=5
base=Path.home()/'inkling-live-bin-p9-72fd097'
for version,lib in [('p9',base),('narrow',Path('build/p11/narrow-bin'))]:
    kernels=configs.copy()
    if version=='narrow':
        kernels={}
        for n in [5,6,7,8]:
            ref=next(k for k in configs if k.startswith('_Z10ik_q2k_moeILi8'))
            kernels[ref.replace('ILi8','ILi'+str(n))]=(f'ik_q2k_moe<{n}>','-','-','-')
    result=subprocess.run([str(Path.home()/'opt/cuda-12.8/bin/cuobjdump'),'--dump-sass','--function',','.join(kernels),str(lib/'libggml-cuda.so.1')],check=True,text=True,capture_output=True)
    opcodes=collections.defaultdict(collections.Counter)
    current=None
    for line in result.stdout.splitlines():
        m=re.search(r'Function : (\S+)',line)
        if m: current=m[1]
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]+)',line)
        if m and current in kernels:opcodes[current][m[1]]+=1
    assert all(opcodes[n] for n in kernels),[(n,opcodes[n]) for n in kernels]
    with (out/f'{version}-sm86-instructions.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(['kernel','observed registers/thread','observed static shared bytes','observed dynamic shared bytes','static matrix/dot-product instructions'])
        for n,(label,regs,stat,dyn) in kernels.items():
            relevant=' '.join(f'{op}:{num}' for op,num in sorted(opcodes[n].items()) if op.startswith(('IMMA','HMMA','DMMA','BMMA','IDP','DP4A')))
            w.writerow([label,regs,stat,dyn,relevant or '(none)'])
