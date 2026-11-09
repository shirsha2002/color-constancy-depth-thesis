"""Resumable paper-budget PCCC runs with separately labelled selection rules.

The primary endpoint is the predeclared final epoch. The author's repeated
held-out-fold checkpoint selection is retained only as an optimistic reference.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import psutil
import torch
from model import ROOT,load,make,forward,synchronize

HOME=ROOT/'phase5_pccc'
sys.path.insert(0,str(ROOT/'phase4_pccc'))
import pilot as old
import upstream

VARIANTS={'weighted_real':('weighted','real',.15),'weighted_flat':('weighted','flat',.15),
          'released_real':('released','real',.2),'released_flat':('released','flat',.2)}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n');temp.replace(path)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial');torch.save(value,temp);temp.replace(path)

def config(args):
    source=[Path(__file__),HOME/'model.py',ROOT/'phase4_pccc/pilot.py',ROOT/'phase4_pccc/upstream.py']
    cfg={'scope':'DepthAWB paper-budget reproduction at256points; not allpaperdatasets/resolutions',
        'client_date':'2026-11-09','manifest_path':args.manifest.relative_to(ROOT).as_posix(),
        'manifest_sha256':sha(args.manifest),'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in source},
        'upstream':upstream.identities(),'device':args.device,'torch':torch.__version__,
        'threads':8,'epochs':args.epochs,'batch_size':16,'learning_rate':.0003,'optimizer':'Adam default parameters',
        'variants':VARIANTS,'folds':[0,1,2],'seed_rule':'20261109+officialfold',
        'primary_checkpoint':'Fixed final epoch; no held-out selection or early stopping',
        'author_reference_checkpoint':'Lowest repeatedly observed held-out-fold angular loss; explicitly test-selected and optimistic',
        'test_labels_in_optimizer':False,'metric':'Recovery angular degrees, float64 score with original cosine clamp0.999999',
        'mixed_precision':False,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'checkpoint_interval':50,
        'geometry_control':'ValidXYZ/abs(Z) to planeZ=-1; keep nativeRGB, rays and invalidXYZ0; not pureRGB or exact fullRGBD rebuild',
        'known_prior_data_access':'Allfolds appeared in the earlier adapted pilot. Reproduction, not independent newconfirmation.',
        'scope_limitations':['Original published code has3-channel sum versus4-channel weighted head in paper.',
            'Families also differ in intensity SD. Head comparison is not isolated.',
            'One predeclared training seed per official fold, not a seed-variance study.',
            'Physical room independence and native sensor registration cannot be verified from sparse release alone.']}
    cfg['sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();return cfg

def rng_state(device,order):
    state=np.random.get_state()
    out={'python':random.getstate(),'numpy_name':state[0],'numpy_keys':torch.from_numpy(state[1].copy()),
        'numpy_pos':state[2],'numpy_has_gauss':state[3],'numpy_cached_gauss':state[4],
        'torch_cpu':torch.get_rng_state(),'order':copy.deepcopy(order.bit_generator.state)}
    if device=='xpu':out['torch_xpu']=torch.xpu.get_rng_state().cpu()
    return out

def restore_rng(state,device,order):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'],state['numpy_keys'].numpy(),state['numpy_pos'],state['numpy_has_gauss'],state['numpy_cached_gauss']))
    torch.set_rng_state(state['torch_cpu']);order.bit_generator.state=state['order']
    if device=='xpu':torch.xpu.set_rng_state(state['torch_xpu'])

def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def batch(ids,values,geometry,device,training=False,rotator=None,sigma=.15):
    x,y=old.tensor_batch(ids,values,geometry,training,rotator,sigma)
    return x.to(device),y.to(device)

def score(net,head,ids,values,geometry,device):
    net.eval();vectors=[];labels=[];author_errors=[]
    with torch.inference_mode():
        for start in range(0,len(ids),16):
            x,y=batch(ids[start:start+16],values,geometry,device)
            pred=forward(net,x,head)
            dot=(torch.nn.functional.normalize(pred,dim=1)*torch.nn.functional.normalize(y,dim=1)).sum(1)
            e=torch.rad2deg(torch.acos(dot.clamp(-.999999,.999999)))
            vectors.extend(pred.cpu().numpy());labels.extend(y.cpu().numpy());author_errors.extend(e.cpu().numpy().tolist())
    return np.stack(vectors),np.stack(labels),float(np.mean(author_errors))

def keep_awake():
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

def train_job(out,cfg,manifest,variant,fold,stop_after=None):
    if cfg['device']=='xpu':assert torch.xpu.is_available(),'Intel XPU unavailable'
    job=out/f'{variant}_fold{fold}';job.mkdir(parents=True,exist_ok=True)
    lock=job/'running.lock'
    if lock.exists():
        owner=json.loads(lock.read_text())
        try:alive=abs(psutil.Process(owner['pid']).create_time()-owner['created'])<1
        except psutil.NoSuchProcess:alive=False
        if alive:raise RuntimeError(f'Job already running: {job.name}, PID{owner["pid"]}')
        lock.unlink()
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'created':psutil.Process().create_time()},f)
    try:return _train_job(job,cfg,manifest,variant,fold,stop_after)
    finally:
        if lock.exists():lock.unlink()

def _train_job(job,cfg,manifest,variant,fold,stop_after):
    for path,expected in cfg['sources'].items():assert sha(ROOT/path)==expected,'Frozen source changed: '+path
    assert upstream.identities()==cfg['upstream']
    done=job/'complete.json'
    if done.exists():
        result=json.loads(done.read_text());assert result['configuration_sha256']==cfg['sha256']
        for name,expected in result['checkpoint_sha256'].items():assert sha(job/'checkpoints'/name)==expected
        return result
    values={}
    for r in manifest['samples']:
        for field in ['cloud','label']:assert sha(ROOT/r[field])==r[field+'_sha256']
        values[r['id']]=(np.load(ROOT/r['cloud']).astype(np.float32),np.load(ROOT/r['label']).astype(np.float32))
    ids=manifest['folds'][str(fold)]['train_ids'];test_ids=manifest['folds'][str(fold)]['test_ids']
    head,geometry,sigma=cfg['variants'][variant];device=cfg['device'];seed=20261109+fold
    author,data,utils=load();rotator=data.PcdColor.__new__(data.PcdColor)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='xpu':torch.xpu.manual_seed_all(seed)
    net=make(author,head,device);optimizer=torch.optim.Adam(net.parameters(),lr=.0003)
    order=np.random.default_rng(seed);history=[];start_epoch=0;elapsed_before=0.;best=float('inf');best_epoch=None;best_state=None
    checkpoint=job/'checkpoints/last.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['configuration_sha256']==cfg['sha256'] and saved['variant']==variant and saved['fold']==fold
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];start_epoch=saved['epoch'];elapsed_before=saved['elapsed_seconds']
        best=saved['best_reference_loss'];best_epoch=saved['best_reference_epoch'];best_state=saved['best_reference_state']
        restore_rng(saved['rng'],device,order)
    keep_awake();start=time.perf_counter()
    print(f'START {job.name} on{device}, epoch{start_epoch}/{cfg["epochs"]}, train{len(ids)}/heldout{len(test_ids)}',flush=True)
    def progress(epoch,status,train_loss):
        elapsed=elapsed_before+time.perf_counter()-start
        dump(job/'progress.json',{'status':status,'variant':variant,'fold':fold,'device':device,
            'epoch':epoch,'target_epochs':cfg['epochs'],'elapsed_seconds':elapsed,'updated_utc':utc(),
            'training_loss_deg':train_loss,'configuration_sha256':cfg['sha256'],
            'remaining_seconds_estimate':elapsed/max(epoch,1)*(cfg['epochs']-epoch),
            'pid':os.getpid(),'selection_note':'Held-out-selected reference is optimistic; primary is fixed final epoch.'})
    for epoch in range(start_epoch+1,cfg['epochs']+1):
        net.train();loss_sum=0.;seen=0;permutation=[ids[i] for i in order.permutation(len(ids))]
        for k in range(0,len(permutation),16):
            x,y=batch(permutation[k:k+16],values,geometry,device,True,rotator,sigma)
            optimizer.zero_grad(set_to_none=True);loss=utils.get_angular_loss(forward(net,x,head),y)
            assert torch.isfinite(loss),'Non-finite training loss'
            loss.backward();optimizer.step();loss_sum+=float(loss.detach().cpu())*len(x);seen+=len(x)
        _,_,reference_loss=score(net,head,test_ids,values,geometry,device)
        history.append({'epoch':epoch,'training_loss_deg':loss_sum/seen,'author_reference_heldout_loss_deg':reference_loss})
        if reference_loss<best:
            best=reference_loss;best_epoch=epoch;best_state=cpu_state(net)
        elapsed=elapsed_before+time.perf_counter()-start
        if epoch%50==0 or epoch==cfg['epochs'] or epoch==stop_after:
            save(checkpoint,{'configuration_sha256':cfg['sha256'],'variant':variant,'fold':fold,'epoch':epoch,
                'model':cpu_state(net),'optimizer':optimizer.state_dict(),'rng':rng_state(device,order),
                'history':history,'elapsed_seconds':elapsed,'best_reference_loss':best,
                'best_reference_epoch':best_epoch,'best_reference_state':best_state})
        if epoch==1 or epoch%10==0:progress(epoch,'running',loss_sum/seen)
        if epoch==1 or epoch%100==0:
            print(f'{job.name}: epoch{epoch}/{cfg["epochs"]}, training{loss_sum/seen:.3f}deg, elapsed{elapsed/60:.1f}min',flush=True)
        if stop_after is not None and epoch>=stop_after and epoch<cfg['epochs']:
            progress(epoch,'checkpointed_for_resume',loss_sum/seen);return {'status':'checkpointed_for_resume','epoch':epoch}
    final_state=cpu_state(net)
    for selection,state,selected_epoch in [('fixed_final',final_state,cfg['epochs']),('author_test_selected',best_state,best_epoch)]:
        save(job/'checkpoints'/f'{selection}.pt',{'configuration_sha256':cfg['sha256'],'variant':variant,
            'fold':fold,'selected_epoch':selected_epoch,'selection':selection,'state_dict':state})
        net.load_state_dict(state);p,y,_=score(net,head,test_ids,values,geometry,device)
        payload={'ids':test_ids,'prediction':p.tolist(),'label':y.tolist(),'selected_epoch':selected_epoch,
            'selection':selection,'angular_error_deg':old.angles(p,y).tolist(),'configuration_sha256':cfg['sha256']}
        dump(job/f'{selection}_predictions.json',payload)
    dump(job/'history.json',history)
    complete={'status':'complete','configuration_sha256':cfg['sha256'],'variant':variant,'fold':fold,
        'device':device,'epochs':cfg['epochs'],'author_reference_selected_epoch':best_epoch,
        'training_seconds':elapsed_before+time.perf_counter()-start,'completed_utc':utc(),
        'checkpoint_sha256':{n:sha(job/'checkpoints'/n) for n in ['fixed_final.pt','author_test_selected.pt']}}
    dump(done,complete);progress(cfg['epochs'],'complete',history[-1]['training_loss_deg'])
    print('COMPLETE '+job.name,flush=True);return complete

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','train'])
    parser.add_argument('--device',choices=['cpu','xpu'],default='cpu');parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--manifest',type=Path,default=HOME/'manifest.json')
    parser.add_argument('--job',default='weighted_real:0');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();args.out=args.out.resolve();args.manifest=args.manifest.resolve()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg=config(args);path=args.out/'config.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(cfg)),'Frozen configuration changed'
    else:dump(path,cfg)
    if args.stage=='init':print(json.dumps({'configuration':'FROZEN','sha256':cfg['sha256'],'device':args.device}));return
    variant,fold=args.job.split(':');assert variant in VARIANTS and int(fold) in [0,1,2]
    manifest=json.loads(args.manifest.read_text());assert sha(args.manifest)==cfg['manifest_sha256']
    train_job(args.out,cfg,manifest,variant,int(fold),args.stop_after_epoch)

if __name__=='__main__':main()
