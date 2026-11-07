"""Frozen-protocol real+simulated CC/depth study on new scene-separated data.

Run cache, train and evaluate as separate resumable stages. Raw measured depth
is read only for supervised training or numerical evaluation; the model forward
receives RGB, cached encoder tokens and the frozen model's own prediction.
"""
from __future__ import annotations
import argparse
import csv
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from science import srgb_to_linear,linear_to_srgb
sys.path.insert(0,str(ROOT/'phase2/mono'))
from cc_model import frozen_model
from model import GeometryChromaticAdapter

SEED=20261107
VARIANTS={
 'ordinary_aug':('ordinary',0.),
 'achromatic':('achromatic',0.),
 'content':('content',0.),
 'chromatic_noeq':('chromatic',0.),
 'chromatic_eq':('chromatic',.2),
 'flat_geometry_eq':('flat_geometry_chromatic',.2),
 'geometry_chromatic_eq':('geometry_chromatic',.2),
}
MANIFESTS={'nyu':'phase3/data/nyu_rgbd/manifest.json','remid':'phase3/data/remid/manifest.json',
           'vari':'phase2/data/vari/manifest.json'}
EVAL_CONDITIONS=['original','control','chroma_rb','chroma_green','exposure']
CHROMA={'chroma_rb':np.exp([.35,0.,-.35]),'chroma_green':np.exp([-.2,.35,-.15])}
LUMA=np.array([.2126,.7152,.0722],dtype=np.float64)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8',newline='\n')
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def inputs():
    result={}
    for family,relative in MANIFESTS.items():
        payload=json.loads((ROOT/relative).read_text(encoding='utf-8'))
        records=[]
        for source in payload['images']:
            if family=='vari' and source['split']!='evaluation':continue
            record=dict(source);record['family']=family
            record['key']=family+'__'+record['id'].replace('/','__')
            record.setdefault('view_group',record['scene'])
            record.setdefault('lighting','original')
            record.setdefault('depth_scale_to_meters',1.)
            records.append(record)
        result[family]=records
    return result

def conditions(record):
    if record['family']!='nyu':return ['original']
    if record['split']=='training':return ['original','train_chroma','train_exposure']
    if record['split']=='validation':return ['original','chroma_rb','exposure']
    return EVAL_CONDITIONS

def chromatic_gain(linear,gain):
    before=float(np.mean(linear@LUMA));after=float(np.mean((linear*gain)@LUMA))
    return np.asarray(gain,dtype=np.float64)*(before/max(after,1e-12))

def corrupt(rgb,condition,image_key):
    """Matched headroom prevents clipping in the causal diagnostic arms.

    Both chromatic changes preserve the source mean linear luminance. The same
    source-only headroom is applied to the neutral and exposure controls.
    """
    linear=srgb_to_linear(rgb).astype(np.float64)
    if condition=='original':return rgb.astype(np.float32),{'log_gain':[0.,0.], 'clipped_fraction':0.,'headroom':1.}
    if condition in ['train_chroma','train_exposure']:
        number=int.from_bytes(hashlib.sha256(image_key.encode()).digest()[:4],'little')
        rng=np.random.default_rng(SEED+number)
        if condition=='train_chroma':
            logs=rng.uniform(-.45,.45,3);logs-=logs.mean();gain=chromatic_gain(linear,np.exp(logs))
        else:gain=np.repeat(float(rng.uniform(.65,1.35)),3)
        headroom=min(1.,.995/max(float((linear*gain).max()),1e-12))
    else:
        all_gains={k:chromatic_gain(linear,v) for k,v in CHROMA.items()}
        maximum=max(float(linear.max()),float((linear*1.3).max()),
                    *(float((linear*v).max()) for v in all_gains.values()))
        headroom=min(1.,.995/max(maximum,1e-12))
        gain=all_gains.get(condition,np.repeat(1.3 if condition=='exposure' else 1.,3))
    changed=linear*headroom*gain
    clipped=float(np.any((changed<0)|(changed>1),axis=2).mean())
    assert clipped==0.,'Predeclared gain/headroom should prevent clipping'
    delta=[float(np.log(gain[0]/gain[1])),float(np.log(gain[2]/gain[1]))]
    return linear_to_srgb(changed.astype(np.float32)),{'log_gain':delta,'gain':gain.tolist(),
        'headroom':headroom,'clipped_fraction':clipped,
        'mean_linear_luminance':float(np.mean(changed@LUMA))}

def config(args):
    source_files=[Path(__file__),Path(__file__).with_name('model.py'),ROOT/'scripts/science.py',
        ROOT/'external/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py',
        ROOT/'external/Depth-Anything-V2/metric_depth/depth_anything_v2/util/transform.py']
    cfg={'seed':SEED,'input_size':args.input_size,'threads':args.threads,'training_steps':args.steps,
        'learning_rate':.001,'weight_decay':.0001,'paired_depth_consistency_weight':.1,
        'regularization_weight':.001,'variant_mode_equivariance_weight':VARIANTS,
        'checkpoint_sha256':sha(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'),
        'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in source_files},
        'upstream_revision':subprocess.check_output(['git','-C',str(ROOT/'external/Depth-Anything-V2'),'rev-parse','HEAD'],text=True).strip(),
        'manifests':{p:sha(ROOT/p) for p in MANIFESTS.values()},
        'evaluation_depth_m':'finite and strictly0.1<GT<10; predictionsclipped[0.1,10]',
        'nyu_crop':'Eigen rows45:471 cols41:601 scaled from480x640',
        'other_crop':'full native frame; zero/invalid measured depth excluded',
        'test_scale_shift_fit':False,'remid_role':'externalevaluationonly',
        'test_selection':'NYU128newscenes; ReMIDall735images/50capturegroups; Variprior264images rendered diagnostic',
        'caveat':'Prototype using established components, not certified novelty or physical illuminant recovery'}
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    cfg['configuration_sha256']=fingerprint
    return cfg

def cache_spec(cfg):
    payload={'checkpoint':cfg['checkpoint_sha256'],'input_size':cfg['input_size'],
        'encoder_source':cfg['sources']['external/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py'],
        'upstream_revision':cfg['upstream_revision'],
        'encoder_transform_source':cfg['sources']['external/Depth-Anything-V2/metric_depth/depth_anything_v2/util/transform.py'],
        'transforms':sha(ROOT/'scripts/science.py'),'recipe':inspect.getsource(corrupt)+inspect.getsource(chromatic_gain),
        'rgb_stats':'64x64 area-resized linearized sRGB','frozen_depth':'uncorrected model predicted camera depth resized to patch grid'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

class Store:
    def __init__(self,model,cfg):
        self.model=model;self.cfg=cfg;self.fingerprint=cache_spec(cfg)
        self.folder=ROOT/'phase3/cache'/self.fingerprint[:16];self.folder.mkdir(parents=True,exist_ok=True)
    def path(self,r,c):return self.folder/f'{r["key"]}__{c}.pt'
    def get(self,r,c='original'):
        p=self.path(r,c)
        if not p.exists():self.make(r,c)
        item=torch.load(p,map_location='cpu',weights_only=True)
        assert item['cache_sha256']==self.fingerprint and item['rgb_sha256']==r['rgb_sha256']
        return item
    def make(self,r,c):
        path=self.path(r,c)
        if path.exists():
            item=torch.load(path,map_location='cpu',weights_only=True)
            assert item['cache_sha256']==self.fingerprint and item['rgb_sha256']==r['rgb_sha256'];return
        assert sha(ROOT/r['rgb'])==r['rgb_sha256'],'RGB changed since manifest'
        source=np.asarray(Image.open(ROOT/r['rgb']).convert('RGB'),dtype=np.float32)/255
        rgb,intervention=corrupt(source,c,r['key'])
        bgr=np.round(rgb[:,:,::-1]*255).astype(np.uint8)
        image,original_hw=self.model.image2tensor(bgr,self.cfg['input_size'])
        assert image.device.type=='cpu'
        with torch.no_grad():
            old=ROOT/'phase2/results/mono/features'/f'{r["id"]}__none.pt'
            if r['family']=='vari' and c=='original' and old.exists():
                prior=torch.load(old,map_location='cpu',weights_only=True)
                assert prior['rgb_sha256']==r['rgb_sha256'] and prior['input_size']==self.cfg['input_size']
                assert prior['base_checkpoint_sha256']==self.cfg['checkpoint_sha256']
                features=prior['features']
            else:features=self.model.pretrained.get_intermediate_layers(image,self.model.intermediate_layer_idx[self.model.encoder],return_class_token=True)
            features=[(a.detach().cpu().contiguous(),b.detach().cpu().contiguous()) for a,b in features]
            patch_hw=(int(image.shape[-2]//14),int(image.shape[-1]//14))
            base=self.model.depth_head(features,*patch_hw)[:,0]*self.model.max_depth
            base_small=F.interpolate(base[:,None],size=patch_hw,mode='bilinear',align_corners=False)[:,0].clamp_min(.001)
        linear=srgb_to_linear(np.round(rgb*255)/255)
        color=torch.from_numpy(cv2.resize(linear,(64,64),interpolation=cv2.INTER_AREA).transpose(2,0,1).copy())[None].float()
        item={'cache_sha256':self.fingerprint,'rgb_sha256':r['rgb_sha256'],'features':features,
            'image_hw':list(image.shape[-2:]),'patch_hw':list(patch_hw),'original_hw':list(original_hw),
            'rgb_linear':color,'base_depth':base_small,'intervention':intervention}
        temp=path.with_suffix('.partial');torch.save(item,temp);temp.replace(path)

@lru_cache(maxsize=220)
def gt_array(path,scale):return np.load(ROOT/path).astype(np.float32)*scale
def target(r):return gt_array(r['depth'],float(r['depth_scale_to_meters']))
def valid_mask(gt,family):
    valid=np.isfinite(gt)&(gt>.1)&(gt<10.)
    if family=='nyu':
        h,w=gt.shape;crop=np.zeros_like(valid)
        crop[round(45*h/480):round(471*h/480),round(41*w/640):round(601*w/640)]=True
        valid&=crop
    assert valid.sum()>=100
    return valid
def score(pred,gt,family):
    mask=valid_mask(gt,family);p=np.asarray(pred,dtype=np.float64)
    assert np.isfinite(p).all()
    p=np.clip(p[mask],.1,10);g=gt[mask].astype(np.float64);d=p-g
    return {'abs_rel':float(np.mean(np.abs(d)/g)),'rmse_m':float(np.sqrt(np.mean(d*d))),
        'delta1':float(np.mean(np.maximum(p/g,g/p)<1.25)),'valid_pixels':int(mask.sum())}

def join(items):
    assert len({tuple(i['patch_hw']) for i in items})==1
    return [(torch.cat([i['features'][layer][0] for i in items]),torch.cat([i['features'][layer][1] for i in items])) for layer in range(4)]
def infer(model,adapter,items,mode):
    features=join(items);aux=None
    if adapter is not None:
        features,aux=adapter(features,torch.cat([i['rgb_linear'] for i in items]),
            torch.cat([i['base_depth'] for i in items]),tuple(items[0]['patch_hw']),mode)
    pred=model.depth_head(features,*items[0]['patch_hw'])[:,0]*model.max_depth
    return pred,aux

def validate(model,adapter,records,store,mode):
    adapter.eval();values=[]
    with torch.no_grad():
        for r in records:
            gt=target(r)
            for c in conditions(r):
                item=store.get(r,c);p,_=infer(model,adapter,[item],mode)
                p=F.interpolate(p[:,None],size=gt.shape,mode='bilinear',align_corners=True)[0,0].numpy()
                values.append(score(p,gt,'nyu')['abs_rel'])
    adapter.train();return float(np.mean(values))

def train(model,records,store,cfg,out):
    training=[r for r in records if r['split']=='training'];validation=[r for r in records if r['split']=='validation']
    assert len(training)==200 and len(validation)==40
    schedule=[];rng=np.random.default_rng(SEED)
    while len(schedule)<cfg['training_steps']:
        for k in rng.permutation(len(training)):
            schedule.append((int(k),'train_chroma' if rng.random()<.8 else 'train_exposure'))
    for variant,(mode,eq_weight) in VARIANTS.items():
        file=out/'models'/f'{variant}.pt';file.parent.mkdir(parents=True,exist_ok=True)
        if file.exists():
            old=torch.load(file,map_location='cpu',weights_only=True);assert old['configuration_sha256']==cfg['configuration_sha256'];continue
        torch.manual_seed(SEED);adapter=GeometryChromaticAdapter();optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf');best_state=None;history=[];start=time.perf_counter()
        for step,(index,condition) in enumerate(schedule[:cfg['training_steps']],1):
            r=training[index];items=[store.get(r,'original'),store.get(r,condition)]
            pred,aux=infer(model,adapter,items,mode)
            reference=torch.from_numpy(target(r))[None,None]
            mask=torch.from_numpy(valid_mask(target(r),'nyu').astype(np.float32))[None,None]
            reference=F.interpolate(reference,size=pred.shape[-2:],mode='nearest')[:,0].expand(2,-1,-1)
            mask=F.interpolate(mask,size=pred.shape[-2:],mode='nearest')[:,0].expand(2,-1,-1)>0
            supervised=(torch.abs(pred[mask]-reference[mask])/reference[mask]).mean()
            both=mask[0]&mask[1]
            consistency=torch.abs(pred[0][both].clamp_min(.1).log()-pred[1][both].clamp_min(.1).log()).mean()
            delta=torch.tensor(items[1]['intervention']['log_gain'],dtype=torch.float32)
            equivariance=F.smooth_l1_loss(aux['illumination'][1]-aux['illumination'][0],delta)
            loss=supervised+.1*consistency+eq_weight*equivariance+.001*aux['regularization']
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
            row={'step':step,'loss':float(loss.detach()),'depth_loss':float(supervised.detach()),
                'consistency':float(consistency.detach()),'illumination_equivariance':float(equivariance.detach())}
            if step%200==0 or step==cfg['training_steps']:
                value=validate(model,adapter,validation,store,mode);row['validation_abs_rel']=value
                if value<best:
                    best=value;best_step=step;best_state={k:v.detach().clone() for k,v in adapter.state_dict().items()}
                print(f'{variant}: {step}/{cfg["training_steps"]} valAbsRel={value:.5f}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
            history.append(row)
        payload={'variant':variant,'mode':mode,'configuration_sha256':cfg['configuration_sha256'],
            'state_dict':best_state,'selected_step':best_step,'validation_abs_rel':best,
            'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'history':history,'seconds':time.perf_counter()-start}
        torch.save(payload,file);dump(out/f'{variant}_training.json',{k:v for k,v in payload.items() if k!='state_dict'})

def evaluate(model,data,store,cfg,out):
    models={'frozen':None};modes={'frozen':None}
    for variant,(mode,_) in VARIANTS.items():
        payload=torch.load(out/'models'/f'{variant}.pt',map_location='cpu',weights_only=True)
        assert payload['configuration_sha256']==cfg['configuration_sha256']
        adapter=GeometryChromaticAdapter();adapter.load_state_dict(payload['state_dict']);adapter.eval()
        models[variant]=adapter;modes[variant]=mode
    rows=[];start=time.perf_counter()
    for family,records in data.items():
        selected=[r for r in records if r['split']=='evaluation']
        for index,r in enumerate(selected,1):
            gt=target(r)
            for condition in conditions(r):
                item=store.get(r,condition)
                for variant,adapter in models.items():
                    folder=out/'predictions'/family/variant;folder.mkdir(parents=True,exist_ok=True)
                    file=folder/f'{r["key"]}__{condition}.npy'
                    if file.exists():pred=np.load(file)
                    else:
                        with torch.no_grad():
                            p,_=infer(model,adapter,[item],modes[variant])
                            pred=F.interpolate(p[:,None],size=gt.shape,mode='bilinear',align_corners=True)[0,0].numpy().astype(np.float32)
                        temp=file.with_suffix('.partial')
                        with temp.open('wb') as handle:np.save(handle,pred)
                        temp.replace(file)
                    rows.append({'family':family,'id':r['id'],'scene':r['scene'],'view_group':r['view_group'],
                        'lighting':r['lighting'],'condition':condition,'method':variant,**score(pred,gt,family)})
            if index%16==0 or index==len(selected):
                csvwrite(out/'metrics.csv',rows)
                print(f'Evaluate {family}: {index}/{len(selected)}, rows{len(rows)}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
    groups={}
    for row in rows:groups.setdefault((row['family'],row['condition'],row['method']),[]).append(row)
    summaries=[]
    for (family,condition,method),values in groups.items():
        summaries.append({'family':family,'condition':condition,'method':method,'n_images':len(values),
            **{metric:float(np.mean([r[metric] for r in values])) for metric in ['abs_rel','rmse_m','delta1']}})
    csvwrite(out/'summary.csv',summaries)
    dump(out/'summary.json',{'configuration_sha256':cfg['configuration_sha256'],'rows':len(rows),'summary':summaries,
        'warning':'Per-image descriptive scores. ReMID has50captures but unknown physicalroomidentities; separate scenecluster analysis required. No novelmethodclaim.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cache','train','evaluate','all'])
    parser.add_argument('--input-size',type=int,default=224);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--steps',type=int,default=600);parser.add_argument('--out',type=Path,default=ROOT/'phase3/results/study_v1')
    parser.add_argument('--families',default='nyu,remid,vari');args=parser.parse_args()
    torch.set_num_interop_threads(1);cfg=config(args);data=inputs();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not subprocess.check_output(['git','-C',str(ROOT/'external/Depth-Anything-V2'),'diff','--name-only'],text=True).strip()
    checked=set()
    for records in data.values():
        for record in records:
            for field in ['rgb','depth']:
                path=record[field]
                if path not in checked:
                    assert sha(ROOT/path)==record[field+'_sha256'],f'Data changed since manifest: {path}'
                    checked.add(path)
    print(f'Input integrity verified: {len(checked)} unique RGB/depth files',flush=True)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==json.loads(json.dumps(cfg)),'Changed numeric config; choose a fresh output directory'
    else:dump(out/'config.json',cfg)
    data={k:v for k,v in data.items() if k in args.families.split(',')}
    model=frozen_model(args.threads);assert not any(p.requires_grad for p in model.parameters());store=Store(model,cfg)
    if args.stage in ['cache','all']:
        total=sum(len(conditions(r)) for records in data.values() for r in records);done=0;start=time.perf_counter()
        for family,records in data.items():
            for r in records:
                for c in conditions(r):
                    store.make(r,c);done+=1
                    if done%32==0 or done==total:print(f'Cache: {done}/{total}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
    if args.stage in ['train','all']:train(model,data['nyu'],store,cfg,out)
    if args.stage in ['evaluate','all']:evaluate(model,data,store,cfg,out)

if __name__=='__main__':main()
