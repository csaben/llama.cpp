import json,os,subprocess,sys,time,urllib.request
sys.path.insert(0,os.path.expanduser('~/llama.cpp-fast/perf'))
label=sys.argv[1]; extra=sys.argv[2:]
H=os.path.expanduser('~'); B=H+'/llama.cpp-fast/build/bin'
R=H+'/llama.cpp-fast'
SNIP=open(R+'/perf/bench.py').read()+"\n"+open(R+'/perf/kernsum.py').read()
src=open(R+'/perf/spec_probe.py').read()
blk=src[src.index('PROMPTS = ['):]
blk=blk[:blk.index('\n]\n')+3]
g={'SNIP':SNIP}; exec(blk,g)
prompts=[p for p in g['PROMPTS'] if p[0] in os.environ.get('KINDS','code,edit,prose').split(',')]
cmd=[B+'/llama-server','-m',H+'/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf','-ngl','99','-fa','on','-ctk','q8_0','-ctv','q8_0','-c','16384','-np','1','--backend-sampling','--host','127.0.0.1','--port','8077']+extra
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',LD_LIBRARY_PATH=B)
p=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    for i in range(240):
        try: urllib.request.urlopen('http://127.0.0.1:8077/health',timeout=1); break
        except Exception: time.sleep(1)
    for kind,prompt in prompts:
        body={'messages':[{'role':'user','content':prompt}],'max_tokens':int(os.environ.get('MAXT','384')),'temperature':0,'seed':1,'cache_prompt':False}
        r=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8077/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=1800))
        m=r['choices'][0]['message']; t=r.get('timings',{})
        open(f'{os.environ["EQ"]}/{label}.{kind}.json','w').write(json.dumps(m,sort_keys=True))
        print(label,kind,'fields',{k:len(v) for k,v in m.items() if isinstance(v,str)},'tok/s',round(t.get('predicted_per_second',0),1),'draft',t.get('draft_n_accepted'),'/',t.get('draft_n'))
finally:
    p.terminate(); p.wait()
