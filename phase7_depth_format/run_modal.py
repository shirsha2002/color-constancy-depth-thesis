"""One bounded T4 experiment. Does nothing remotely until explicitly invoked."""
from pathlib import Path
import modal

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/root/proj'
RUN = 'nyu_frame_v1'
CONDITIONS = ['camera_real', 'camera_unitdepth', 'camera_flat']
image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
         .pip_install('torch==2.5.1', 'numpy<2.3', 'pandas', 'psutil', 'scipy', 'opencv-python-headless')
         .add_local_dir(LOCAL/'phase4_pccc', f'{REMOTE}/phase4_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'phase5_pccc', f'{REMOTE}/phase5_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'external/Point-Cloud-Color-Constancy', f'{REMOTE}/external/Point-Cloud-Color-Constancy')
         .add_local_dir(LOCAL/'phase7_depth_format', f'{REMOTE}/phase7_depth_format', ignore=['results/**','__pycache__/**'])
         .add_local_dir(LOCAL/'phase3/data/pccc/raw/NYU_DIODE_label/all_label',
                        f'{REMOTE}/phase3/data/pccc/raw/NYU_DIODE_label/all_label'))
volume = modal.Volume.from_name('pccc-results', create_if_missing=False)
app = modal.App('pccc-depth-format-v1', image=image)


def setup():
    import os, sys, subprocess
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '4'
    for key, value in [('safe.directory','*'), ('core.autocrlf','true'), ('core.fileMode','false')]:
        subprocess.run(['git','config','--global','--add',key,value],check=True)
    sys.path[:0] = [f'{REMOTE}/phase5_pccc', f'{REMOTE}/phase4_pccc']
    import torch
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name()
    return torch


@app.function(gpu='T4', cpu=(2,4), memory=(2048,4096), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def smoke():
    """Training-only timing, finite gradients; no held-out labels or selection."""
    import json, random, sys, time
    import numpy as np
    torch = setup()
    from model import load,make,forward
    import pilot
    manifest = json.loads(Path(f'{REMOTE}/phase7_depth_format/inputs/camera_unitdepth/manifest.json').read_text())
    ids = manifest['folds']['0']['train_ids']
    records = {r['id']:r for r in manifest['samples']}
    values = {i:(np.load(f"{REMOTE}/{records[i]['cloud']}").astype(np.float32),
                 np.load(f"{REMOTE}/{records[i]['label']}").astype(np.float32)) for i in ids}
    author,data,utils = load();rot=data.PcdColor.__new__(data.PcdColor)
    random.seed(20261109);np.random.seed(20261109);torch.manual_seed(20261109)
    net=make(author,'released','cuda');opt=torch.optim.Adam(net.parameters(),lr=.0003)
    times=[];order=np.random.default_rng(20261109)
    for epoch in range(40):
        torch.cuda.synchronize();start=time.perf_counter();net.train()
        perm=[ids[i] for i in order.permutation(len(ids))]
        for k in range(0,len(perm),16):
            x,y=pilot.tensor_batch(perm[k:k+16],values,'real',True,rot,.2)
            opt.zero_grad(set_to_none=True)
            loss=utils.get_angular_loss(forward(net,x.cuda(),'released'),y.cuda())
            assert torch.isfinite(loss)
            loss.backward();opt.step()
        torch.cuda.synchronize();times.append(time.perf_counter()-start)
    result={'status':'PASS','gpu':str(torch.cuda.get_device_name()),'torch':str(torch.__version__),
            'training_only':True,'epochs':40,'train_samples':len(ids),'threads':4,
            'median_train_seconds_per_epoch':float(np.median(times[5:])),
            'estimated_nine_job_hours_with_50percent_margin':float(np.median(times[5:])*10000*9*1.5/3600),
            'last_training_loss':float(loss.detach().cpu()),'pid':__import__('os').getpid()}
    return result


@app.function(gpu='T4', cpu=(2,4), memory=(2048,4096), timeout=3000,
              retries=0, max_containers=9, scaledown_window=2, single_use_containers=True,
              volumes={'/out':volume})
def train(condition:str,fold:int):
    import argparse,hashlib,json,sys
    import numpy as np
    torch=setup()
    import train as t5
    original=t5.rng_state
    def rng_state(device,order):
        state=original(device,order)
        state['numpy_keys']=torch.from_numpy(np.random.get_state()[1].astype(np.int64))
        return state
    t5.rng_state=rng_state
    assert condition in CONDITIONS and fold in range(3)
    manifest_path=Path(f'{REMOTE}/phase7_depth_format/inputs/{condition}/manifest.json')
    cfg=t5.config(argparse.Namespace(device='cuda',epochs=10000,manifest=manifest_path))
    cfg.update(scope='NYU_DIODE camera-convention and scale development study; not a new algorithm',
               client_date='2026-11-16',threads=4,condition=condition,
               cloud_gpu=torch.cuda.get_device_name(),cloud_timeout_seconds=3000,
               additional_sources={n:hashlib.sha256(Path(f'{REMOTE}/phase7_depth_format/{n}').read_bytes()).hexdigest()
                                   for n in ['PROTOCOL.md','prepare.py','run_modal.py']})
    cfg.pop('sha256',None)
    cfg['sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    out=Path('/out')/RUN/condition
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'config.json'
    normalized=json.loads(json.dumps(cfg))
    if config_path.exists():
        assert json.loads(config_path.read_text())==normalized
    else:
        t5.dump(config_path,cfg)
    manifest=json.loads(manifest_path.read_text())
    assert t5.sha(manifest_path)==cfg['manifest_sha256']
    result=t5.train_job(out,cfg,manifest,'released_real',fold)
    volume.commit()
    assert result['fold']==fold
    return {**result,'condition':condition}


@app.local_entrypoint()
def main(confirm_budget:float=0,stage:str='smoke'):
    import json,time
    assert confirm_budget==8,'Requires the user-approved US$8 cap; no default cloud launch'
    home=LOCAL/'phase7_depth_format'
    budget=json.loads((home/'budget.json').read_text())
    assert budget['compute_upper_planning_usd']+budget['overhead_reserve_usd']<8
    assert stage in ['smoke','train']
    if stage=='smoke':
        result=smoke.remote()
        (home/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return
    preflight=json.loads((home/'preflight.json').read_text())
    timing=json.loads((home/'smoke.json').read_text())
    assert preflight['status']==timing['status']=='PASS'
    rate=.000164+4*.0000131+4*.00000222
    assert timing['estimated_nine_job_hours_with_50percent_margin']*3600*rate+1.5<8,'Timing exceeds cap'
    calls={f'{c}/{f}':train.spawn(c,f) for c in CONDITIONS for f in range(3)}
    def record(name,value):
        p=home/'launch.json'
        doc=json.loads(p.read_text()) if p.exists() else {'run':RUN,'approved_usd':8,'jobs':{}}
        doc['jobs'][name]=value
        p.write_text(json.dumps(doc,indent=2)+'\n')
    for name,call in calls.items():record(name,{'call_id':call.object_id,'status':'submitted'})
    failures=[]
    while calls:
        for name,call in list(calls.items()):
            try:
                result=call.get(timeout=0)
                record(name,{'status':'complete','result':result})
                print('COMPLETE',name,json.dumps(result),flush=True)
                del calls[name]
            except TimeoutError:
                pass
            except Exception as exc:
                record(name,{'status':'failed','error':repr(exc)})
                print('FAILED',name,repr(exc),flush=True);failures.append(name);del calls[name]
        if calls:time.sleep(10)
    assert not failures,f'Failed jobs require manual review, no automatic retry: {failures}'
    print('ALL NINE MODELS COMPLETE',flush=True)

