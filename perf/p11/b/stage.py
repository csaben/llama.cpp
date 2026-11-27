from pathlib import Path
import argparse, hashlib, os, re, shutil, subprocess, tempfile
repo=Path.cwd()
args=argparse.ArgumentParser(description="Stage one verified runtime and write its manual deployment script.")
args.add_argument("source")
args.add_argument("runtime")
args.add_argument("--full-rate-policy", action="store_true")
args.add_argument("--mma", type=int, choices=(3,7), default=3)
args.add_argument("--window-cols", type=int, choices=(1,2), default=1)
settings=args.parse_args()
source, runtime=settings.source, settings.runtime
runtime=Path(runtime).resolve()
assert subprocess.check_output(['git','rev-parse',source+'^{commit}'],text=True).startswith(source)
base=Path.home()/('inkling-live-bin-p11b-'+source)
names=[s.split()[1] for s in (repo/'perf/p9/snapshot.sha256').read_text().splitlines()]
assert len(names)==9
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
manifest=''.join(f'{digest(runtime / n)}  {n}\n' for n in names)
if not base.exists():
    with tempfile.TemporaryDirectory(dir=base.parent,prefix='.p11b-stage-') as tmp:
        staged=Path(tmp)/'snapshot'
        staged.mkdir()
        for n in names: shutil.copy2(runtime/n,staged/n)
        (staged/'SHA256SUMS').write_text(manifest)
        subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=staged,check=True)
        staged.rename(base)
assert (base/'SHA256SUMS').read_text()==manifest
evidence=repo/'perf/p11/b'
(evidence/'snapshot.sha256').write_text(manifest)
log=subprocess.check_output(['sha256sum','-c',str(evidence/'snapshot.sha256')],cwd=base,text=True)
env={**os.environ,'LD_LIBRARY_PATH':str(base)}
loader=subprocess.check_output(['ldd',str(base/'llama-server')],env=env,text=True)
for n in names:
    if n=='llama-server':continue
    assert re.search(r'^\s*'+re.escape(n)+r' => '+re.escape(str(base/n))+r'\s',loader,re.M),n
(evidence/'snapshot-checks.txt').write_text(log+re.sub(r' \(0x[0-9a-f]+\)', '', loader))
deploy=(repo/'perf/live/deploy-p9v-uncensored-128k.sh').read_text()
deploy=deploy.replace(deploy.splitlines()[1]+'\n'+deploy.splitlines()[2], '# Serve the verified Phase 11 Stream B snapshot on physical GPU0.')
deploy=deploy.replace('inkling-live-bin-p9-72fd097',base.name).replace('perf/p9/snapshot.sha256','perf/p11/b/snapshot.sha256')
deploy=deploy.replace('CUDA_VISIBLE_DEVICES=1','CUDA_VISIBLE_DEVICES=0').replace('Phase 9 vision','Phase 11 Stream B')
deploy=deploy.replace('Restoring phase 5.','Restoring the 128K p9 release.').replace('bash perf/live/restore-p5.sh','bash perf/live/deploy-p9v-uncensored-128k.sh')
deploy=deploy.replace('--setenv=IK_MMA=3', f'--setenv=IK_MMA={settings.mma}')
deploy=deploy.replace('--setenv=IK_WINDOW_COLS=1', f'--setenv=IK_WINDOW_COLS={settings.window_cols}')
if settings.full_rate_policy:
    deploy=deploy.replace('--setenv=LLAMA_DSPARK_POLICY=1', '--setenv=LLAMA_DSPARK_POLICY=1 --setenv=LLAMA_DSPARK_FULL_RATE=1')
for flag in ('--port 8031', '-c 131072', '--alias inkling-mini', '--mmproj "$mmproj"', '--setenv=CUDA_VISIBLE_DEVICES=0',
             'bash perf/live/deploy-p9v-uncensored-128k.sh', f'--setenv=IK_MMA={settings.mma}', f'--setenv=IK_WINDOW_COLS={settings.window_cols}'):
    assert flag in deploy, flag
p=repo/'perf/live/deploy-p11b.sh'
p.write_text(deploy)
p.chmod(0o775)
subprocess.run(['bash','-n',str(p)],check=True)
print(base)
