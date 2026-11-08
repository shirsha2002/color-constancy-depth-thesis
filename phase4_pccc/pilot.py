"""CPU PCCC pilot with genuine released colored clouds and illuminant labels.

Test scoring is a separate stage after all predeclared models finish training.
No checkpoint selection or early stopping uses the official outer test fold.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from upstream import ROOT,load,make_model,predict,identities,sha

HOME=ROOT/'phase4_pccc';SEEDS=[20261109,20261110,20261111]
VARIANTS={'released_real':('released_sum','real',.2),'released_flat':('released_sum','flat',.2),
          'weighted_real':('paper_weighted','real',.15),'weighted_flat':('paper_weighted','flat',.15)}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')

def angles(pred,labels):
    a=np.asarray(pred,dtype=np.float64);b=np.asarray(labels,dtype=np.float64)
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-.999999,.999999)))

def flatten(points):
    x=points.copy();valid=np.abs(x[:,2])>1e-8
    x[valid,:3]/=np.abs(x[valid,2:3])
    # Keep the native negative-Z convention, original camera rays and zero mask.
    assert np.allclose(x[valid,2],-1) and (x[~valid,:3]==0).all()
    assert np.array_equal(x[:,3:],points[:,3:])
    return x

def augment(x,rotator,sigma):
    # Same XYZ permutation / RotateZ as the author loader.
    x=x.copy();p=np.zeros_like(x[:,:3]);p[:,0]=x[:,2];p[:,1]=x[:,0];p[:,2]=x[:,1]
    p=rotator.RotateZ(p);x[:,0]=p[:,1];x[:,1]=p[:,2];x[:,2]=p[:,0]
    x[:,3:]*=np.random.normal(1,sigma)
    return x

def arrays(manifest):
    records=manifest['samples'];values={}
    for r in records:
        for field in ['cloud','label']:assert sha(ROOT/r[field])==r[field+'_sha256']
        values[r['id']]=(np.load(ROOT/r['cloud']).astype(np.float32),np.load(ROOT/r['label']).astype(np.float32))
    return records,values

def configuration(args):
    cfg={'role':'Small CPU adapted PCCC reproduction; not full paper benchmark',
        'manifest_sha256':sha(HOME/'manifest.json'),'upstream':identities(),
        'sources':{p.name:sha(p) for p in [Path(__file__),HOME/'upstream.py']},
        'device':'cpu','threads':args.threads,'epochs':args.epochs,'batch_size':16,
        'learning_rate':.0003,'optimizer':'Adam, author defaults','seeds':SEEDS,'variants':VARIANTS,
        'points_per_sample':256,'checkpoint_rule':'Lowest mean inner-validation recovery angular loss, evaluated every epoch; earliest tie',
        'test_fold':0,'selection_test_access':False,'test_metric':'Recovery angular degrees, double precision, cosine clamped+/-0.999999 as author loss',
        'released_augmentation':'Author RotateZ Gaussian30degrees and RGB scalar sigma0.2',
        'weighted_augmentation':'Same RotateZ; RGB scalar sigma0.15 as paper Eq.6',
        'head_comparison_note':'Families also differ in intensity augmentation; not an isolated head ablation.',
        'flat_geometry':'Divide valid native XYZ by abs(Z), yielding Z=-1 and preserved rays; keep missing XYZ0. Not RGB-only.',
        'paper_budget':'Original paper10000epochs/fullcrossvalidation; this pilot uses a reduced budget and one outerfold.'}
    cfg['sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();return cfg

def tensor_batch(ids,values,geometry,training=False,rotator=None,sigma=.2):
    xs=[];ys=[]
    for key in ids:
        x,y=values[key];x=flatten(x) if geometry=='flat' else x.copy()
        if training:x=augment(x,rotator,sigma)
        xs.append(x.T.copy());ys.append(y)
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))

def validation(model,head,ids,values,geometry):
    model.eval();errors=[]
    with torch.inference_mode():
        for start in range(0,len(ids),16):
            x,y=tensor_batch(ids[start:start+16],values,geometry)
            errors.extend(angles(predict(model,x,head).numpy(),y.numpy()).tolist())
    return float(np.mean(errors))

def train(args,cfg,records,values,author,data,utils):
    train_ids=[r['id'] for r in records if r['split']=='train'];val_ids=[r['id'] for r in records if r['split']=='validation']
    rotator=data.PcdColor.__new__(data.PcdColor)
    out=args.out
    for variant,(head,geometry,sigma) in VARIANTS.items():
        for seed in SEEDS:
            dest=out/'models'/f'{variant}_{seed}.pt'
            if dest.exists():
                checkpoint=torch.load(dest,weights_only=True,map_location='cpu')
                assert checkpoint['configuration_sha256']==cfg['sha256'];continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=make_model(author,head);optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
            history=[];best=float('inf');best_state=None;selected=None;start=time.perf_counter()
            schedule=np.random.default_rng(seed)
            for epoch in range(1,args.epochs+1):
                model.train();losses=[]
                order=[train_ids[i] for i in schedule.permutation(len(train_ids))]
                for k in range(0,len(order),16):
                    x,y=tensor_batch(order[k:k+16],values,geometry,True,rotator,sigma)
                    optimizer.zero_grad(set_to_none=True)
                    loss=utils.get_angular_loss(predict(model,x,head),y)
                    assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss.detach()))
                score=validation(model,head,val_ids,values,geometry)
                row={'epoch':epoch,'training_angular_loss_deg':float(np.mean(losses)),'validation_mean_deg':score}
                history.append(row)
                if score<best:
                    best=score;selected=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if epoch==1 or epoch%10==0:
                    print(f'{variant} seed{seed} epoch{epoch}/{args.epochs}: validation{score:.3f}deg, elapsed{time.perf_counter()-start:.1f}s',flush=True)
            checkpoint={'configuration_sha256':cfg['sha256'],'variant':variant,'head':head,'geometry':geometry,
                'seed':seed,'selected_epoch':selected,'validation_mean_deg':best,'history':history,
                'state_dict':best_state,'trainable_parameters':sum(p.numel() for p in model.parameters()),
                'training_seconds':time.perf_counter()-start}
            dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.partial');torch.save(checkpoint,temp);temp.replace(dest)
            dump(out/'histories'/f'{variant}_{seed}.json',{k:v for k,v in checkpoint.items() if k!='state_dict'})
            print(f'COMPLETE {variant} seed{seed}: selected epoch{selected}, val{best:.3f}deg',flush=True)

def evaluate(args,cfg,records,values,author):
    expected=[args.out/'models'/f'{v}_{s}.pt' for v in VARIANTS for s in SEEDS]
    assert all(p.exists() for p in expected),'All models must finish before any final test scoring'
    checks={p.name:sha(p) for p in expected}
    frozen=args.out/'evaluation_sources.json'
    identity={'configuration_sha256':cfg['sha256'],'checkpoint_sha256':checks}
    if frozen.exists():assert json.loads(frozen.read_text())==identity
    else:dump(frozen,identity)
    ids=[r['id'] for r in records if r['split']=='test'];train_ids=[r['id'] for r in records if r['split']=='train']
    rows=[]
    def record(method,seed,pred,labels):
        for key,p,y,error in zip(ids,pred,labels,angles(pred,labels)):
            rows.append({'id':key,'method':method,'seed':seed,'pred_r':float(p[0]),'pred_g':float(p[1]),'pred_b':float(p[2]),
                'gt_r':float(y[0]),'gt_g':float(y[1]),'gt_b':float(y[2]),'angular_error_deg':float(error)})
    labels=np.stack([values[k][1] for k in ids])
    gw=np.stack([values[k][0][:,3:].mean(0) for k in ids]);record('gray_world_points','none',gw,labels)
    mean=np.stack([values[k][1]/np.linalg.norm(values[k][1]) for k in train_ids]).mean(0)
    record('training_mean_illuminant','none',np.repeat(mean[None],len(ids),axis=0),labels)
    for variant,(head,geometry,sigma) in VARIANTS.items():
        for seed in SEEDS:
            checkpoint=torch.load(args.out/'models'/f'{variant}_{seed}.pt',map_location='cpu',weights_only=True)
            assert checkpoint['configuration_sha256']==cfg['sha256']
            model=make_model(author,head);model.load_state_dict(checkpoint['state_dict']);model.eval();pred=[]
            with torch.inference_mode():
                for start in range(0,len(ids),16):
                    x,_=tensor_batch(ids[start:start+16],values,geometry)
                    pred.extend(predict(model,x,head).numpy())
            record(variant,seed,np.stack(pred),labels)
    path=args.out/'predictions.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(args.out/'evaluation_complete.json',{'rows':len(rows),'test_samples':len(ids),'models':len(expected),
        'predictions_csv_sha256':sha(path),'test_used_for_selection':False})
    print(json.dumps({'evaluation':'COMPLETE','rows':len(rows),'test_samples':len(ids)}),flush=True)

def benchmark(author,data,utils,values):
    ids=list(values)[:16];x,y=tensor_batch(ids,values,'real')
    results=[]
    for threads in [2,4,8]:
        torch.set_num_threads(threads);torch.manual_seed(1);m=make_model(author,'released_sum')
        for _ in range(2):m.zero_grad(set_to_none=True);utils.get_angular_loss(predict(m,x,'released_sum'),y).backward()
        start=time.perf_counter()
        for _ in range(5):m.zero_grad(set_to_none=True);utils.get_angular_loss(predict(m,x,'released_sum'),y).backward()
        results.append({'threads':threads,'batch16_forward_backward_seconds':(time.perf_counter()-start)/5})
    dump(HOME/'benchmark.json',{'data':'training samples only','measurements':results})
    print(json.dumps(results,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['benchmark','init','train','evaluate'])
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--out',type=Path,default=HOME/'results/cpu_v1');args=parser.parse_args()
    torch.set_num_interop_threads(1);torch.set_num_threads(args.threads)
    author,data,utils=load();manifest=json.loads((HOME/'manifest.json').read_text());records,values=arrays(manifest)
    if args.stage=='benchmark':benchmark(author,data,utils,{k:v for k,v in values.items() if k in {r['id'] for r in records if r['split']=='train'}});return
    cfg=configuration(args);args.out.mkdir(parents=True,exist_ok=True);path=args.out/'config.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(cfg)),'Configuration changed; use a new output directory'
    else:dump(path,cfg)
    if args.stage=='init':print(json.dumps({'protocol_frozen':cfg['sha256']}));return
    if args.stage=='train':train(args,cfg,records,values,author,data,utils)
    else:evaluate(args,cfg,records,values,author)

if __name__=='__main__':main()
