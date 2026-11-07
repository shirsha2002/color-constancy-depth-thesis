"""Validation-motivated local coupling, locked before any phase3 test scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import study
from spatial_model import SpatialGeometryAdapter

VARIANTS={
    'spatial_content':('spatial_content',.2,.01),
    'spatial_flat':('spatial_flat',.2,.01),
    'spatial_no_graph_loss':('spatial_geometry',.2,0.),
    'spatial_no_equivariance':('spatial_geometry',0.,.01),
    'spatial_geometry_eq':('spatial_geometry',.2,.01),
}

def configuration(args):
    cfg=study.config(args);cfg.pop('configuration_sha256')
    cfg.update({'architecture':'local predicted-geometry chromatic graph adapter',
        'variant_mode_equivariance_weight':{k:[v[0],v[1]] for k,v in VARIANTS.items()},
        'graph_loss_weights':{k:v[2] for k,v in VARIANTS.items()},
        'graph_log_depth_bandwidth':.15,'primary_method':'spatial_geometry_eq',
        'local_equivariance':'SmoothL1 local q_aug-q_original versus known2D loggain at each token',
        'graph_penalty':'predicted-depth-affinity weighted local L1 chroma differences',
        'development_note':'Designed after global-prototype validation; locked before any new NYUtest/ReMID score inspection.',
        'trainable_parameters':sum(p.numel() for p in SpatialGeometryAdapter().parameters())})
    for p in [Path(__file__),Path(__file__).with_name('spatial_model.py')]:cfg['sources'][p.relative_to(study.ROOT).as_posix()]=study.sha(p)
    cfg['configuration_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();return cfg

def train(model,data,store,cfg,out):
    records=data['nyu'];training=[r for r in records if r['split']=='training'];validation=[r for r in records if r['split']=='validation']
    rng=np.random.default_rng(study.SEED);schedule=[]
    while len(schedule)<cfg['training_steps']:
        for k in rng.permutation(len(training)):schedule.append((int(k),'train_chroma' if rng.random()<.8 else 'train_exposure'))
    for variant,(mode,eq_weight,graph_weight) in VARIANTS.items():
        file=out/'models'/f'{variant}.pt';file.parent.mkdir(parents=True,exist_ok=True)
        if file.exists():
            payload=torch.load(file,map_location='cpu',weights_only=True);assert payload['configuration_sha256']==cfg['configuration_sha256'];continue
        torch.manual_seed(study.SEED);adapter=SpatialGeometryAdapter();optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.0001)
        history=[];best=float('inf');best_state=None;start=time.perf_counter()
        for step,(index,condition) in enumerate(schedule[:cfg['training_steps']],1):
            r=training[index];items=[store.get(r,'original'),store.get(r,condition)]
            pred,aux=study.infer(model,adapter,items,mode)
            gt=torch.from_numpy(study.target(r))[None,None]
            valid=torch.from_numpy(study.valid_mask(study.target(r),'nyu').astype(np.float32))[None,None]
            gt=F.interpolate(gt,size=pred.shape[-2:],mode='nearest')[:,0].expand(2,-1,-1)
            valid=F.interpolate(valid,size=pred.shape[-2:],mode='nearest')[:,0].expand(2,-1,-1)>0
            depth=(torch.abs(pred[valid]-gt[valid])/gt[valid]).mean();common=valid[0]&valid[1]
            consistency=torch.abs(pred[0][common].clamp_min(.1).log()-pred[1][common].clamp_min(.1).log()).mean()
            delta=torch.tensor(items[1]['intervention']['log_gain']).float()[None].expand_as(aux['local_illumination'][0])
            equivariance=F.smooth_l1_loss(aux['local_illumination'][1]-aux['local_illumination'][0],delta)
            loss=depth+.1*consistency+eq_weight*equivariance+graph_weight*aux['geometry_graph_penalty']+.001*aux['regularization']
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step()
            row={'step':step,'loss':float(loss.detach()),'depth_loss':float(depth.detach()),'consistency':float(consistency.detach()),
                'local_equivariance':float(equivariance.detach()),'geometry_graph_penalty':float(aux['geometry_graph_penalty'].detach())}
            if step%200==0 or step==cfg['training_steps']:
                val=study.validate(model,adapter,validation,store,mode);row['validation_abs_rel']=val
                if val<best:best=val;best_step=step;best_state={k:v.detach().clone() for k,v in adapter.state_dict().items()}
                print(f'{variant}: {step}/{cfg["training_steps"]}, valAbsRel={val:.5f}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
            history.append(row)
        payload={'variant':variant,'mode':mode,'configuration_sha256':cfg['configuration_sha256'],'state_dict':best_state,
            'selected_step':best_step,'validation_abs_rel':best,'trainable_parameters':cfg['trainable_parameters'],
            'history':history,'seconds':time.perf_counter()-start}
        torch.save(payload,file);study.dump(out/f'{variant}_training.json',{k:v for k,v in payload.items() if k!='state_dict'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','train','evaluate'])
    parser.add_argument('--input-size',type=int,default=224);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--steps',type=int,default=600);parser.add_argument('--out',type=Path,default=study.ROOT/'phase3/results/spatial_v1')
    args=parser.parse_args();torch.set_num_interop_threads(1);cfg=configuration(args);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==json.loads(json.dumps(cfg))
    else:study.dump(out/'config.json',cfg)
    if args.stage=='check':
        torch.set_num_threads(2);torch.manual_seed(study.SEED)
        f=[(torch.randn(2,80,384),torch.randn(2,384)) for _ in range(4)];rgb=.01+.5*torch.rand(2,3,64,64);z=1+torch.rand(2,8,10)
        checks={}
        for mode in ['spatial_content','spatial_flat','spatial_geometry']:
            a=SpatialGeometryAdapter();result,aux=a(f,rgb,z,(8,10),mode)
            assert all(torch.equal(x[0],y[0]) for x,y in zip(result,f));assert aux['local_illumination'].shape==(2,80,2)
            loss=sum(t[0].square().mean() for t in result)+aux['local_illumination'].square().mean()+aux['geometry_graph_penalty']
            loss.backward();assert a.task_gamma.grad.abs().sum()>0 and any(p.grad is not None and p.grad.abs().sum()>0 for p in a.up.parameters())
            checks[mode]={'identity':True,'gradient_path':True,'local_chromatic_field':True}
        a=SpatialGeometryAdapter();_,x=a(f,rgb,z,(8,10),'spatial_geometry');_,y=a(f,rgb,3*z,(8,10),'spatial_geometry')
        assert torch.allclose(x['local_illumination'],y['local_illumination'],atol=1e-6)
        study.dump(out/'module_check.json',{'checks':'PASS','modes':checks,'pseudo_geometry_scale_invariance':True,'parameters':cfg['trainable_parameters']})
        print(json.dumps({'check':'PASS','parameters':cfg['trainable_parameters']},indent=2));return
    model=study.frozen_model(args.threads);data=study.inputs();store=study.Store(model,cfg)
    for relative,expected in {**cfg['sources'],**cfg['manifests']}.items():assert study.sha(study.ROOT/relative)==expected
    if args.stage=='train':train(model,data,store,cfg,out)
    else:
        study.GeometryChromaticAdapter=SpatialGeometryAdapter
        study.VARIANTS={k:[v[0],v[1]] for k,v in VARIANTS.items()}
        study.evaluate(model,data,store,cfg,out)

if __name__=='__main__':main()
