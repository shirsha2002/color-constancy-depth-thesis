"""Reproducible scene-held-out exploratory Vari depth/CC integration study."""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from cc_model import ROOT,ColorFeatureAdapter,frozen_model,predict_from_features,invariant_feature_loss
sys.path.insert(0,str(ROOT/'scripts'))
from science import srgb_to_linear,transform

OUT=ROOT/'phase2/results/mono'
VARIANTS=['task_adapter','achromatic_consistency','cc_supervised','cc_consistency']
EVAL_METHODS=['frozen','gray_world_preprocess']+VARIANTS
SEED=20261028

def read_manifest(path):
    payload=json.loads(path.read_text(encoding='utf-8'));records=payload['images']
    needed={'id','scene','view_group','lighting','rgb','depth','split'}
    for r in records:
        if not needed<=r.keys():raise ValueError(f'Missing manifest fields:{needed-r.keys()}')
    rooms={split:{str(r['scene']) for r in records if r['split']==split} for split in ['training','validation','evaluation']}
    if not rooms['training'] or not rooms['validation'] or not rooms['evaluation']:raise ValueError('All three splits required')
    if rooms['evaluation']&(rooms['training']|rooms['validation']):raise ValueError('Evaluation room overlaps development')
    groups={split:{r['view_group'] for r in records if r['split']==split} for split in rooms}
    if groups['training']&groups['validation']:raise ValueError('Camera group crosses training/validation')
    return payload,records

def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def cache_path(record,preprocess='none'):
    return OUT/'features'/f"{record['id']}__{preprocess}.pt"

def descriptor(rgb):
    linear=srgb_to_linear(rgb)
    means=np.maximum(linear.mean(axis=(0,1),dtype=np.float64),1e-5)
    logs=np.log(means);chromatic=logs-logs.mean()
    lum=float(np.mean(linear@np.array([.2126,.7152,.0722],dtype=np.float32)))
    return dict(chromatic=chromatic.astype(np.float32).tolist(),achromatic=[float(np.log(max(lum,1e-5)))]*3)

def make_cache(model,records,input_size,manifest_hash):
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True)
    base_hash=hashfile(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
    total=len(records)+sum(r['split']=='evaluation' for r in records);done=0;start=time.perf_counter()
    for r in records:
        rgb=np.asarray(Image.open(ROOT/r['rgb']).convert('RGB'),dtype=np.float32)/255
        for preprocessing in (['none','gray_world'] if r['split']=='evaluation' else ['none']):
            path=cache_path(r,preprocessing)
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True)
                if old['input_size']!=input_size or old['manifest_sha256']!=manifest_hash or old.get('base_checkpoint_sha256')!=base_hash:raise ValueError('Feature cache configuration changed')
            else:
                processed=rgb if preprocessing=='none' else transform(rgb,'clean','gray_world')[0]
                bgr=np.round(processed[:,:,::-1]*255).astype(np.uint8)
                image,original_hw=model.image2tensor(bgr,input_size)
                with torch.no_grad():
                    feats=model.pretrained.get_intermediate_layers(image,model.intermediate_layer_idx[model.encoder],return_class_token=True)
                item=dict(features=[(a.detach().cpu().contiguous(),b.detach().cpu().contiguous()) for a,b in feats],
                    patch_hw=[int(image.shape[-2]//14),int(image.shape[-1]//14)],
                    image_hw=[int(image.shape[-2]),int(image.shape[-1])],original_hw=list(original_hw),
                    descriptor=descriptor(rgb),input_size=input_size,manifest_sha256=manifest_hash,rgb_sha256=hashfile(ROOT/r['rgb']),base_checkpoint_sha256=base_hash)
                tmp=path.with_suffix('.partial');torch.save(item,tmp);tmp.replace(path)
            done+=1
            if done%16==0 or done==total:print(f'Feature cache {done}/{total},elapsed{time.perf_counter()-start:.0f}s',flush=True)

def load_item(r,preprocess='none'):
    return torch.load(cache_path(r,preprocess),map_location='cpu',weights_only=True)

def join_features(items):
    shapes={tuple(x['patch_hw']) for x in items}
    if len(shapes)!=1:raise ValueError('Paired camera shapes differ')
    return [(torch.cat([x['features'][i][0] for x in items]),torch.cat([x['features'][i][1] for x in items])) for i in range(4)]

def statistics(records):
    items=[load_item(r)['descriptor'] for r in records if r['split']=='training']
    result={}
    for key in ['chromatic','achromatic']:
        x=np.array([a[key] for a in items],dtype=np.float32)
        result[key]={'mean':x.mean(0).tolist(),'std':np.maximum(x.std(0),.05).tolist()}
    return result

def condition(items,variant,stats):
    if variant=='task_adapter':return torch.zeros((len(items),3))
    key='achromatic' if variant=='achromatic_consistency' else 'chromatic'
    x=torch.tensor([r['descriptor'][key] for r in items],dtype=torch.float32)
    return (x-torch.tensor(stats[key]['mean']))/torch.tensor(stats[key]['std'])

def target(r,hw):
    g=torch.from_numpy(np.load(ROOT/r['depth']).astype(np.float32))[None,None]
    return F.interpolate(g,size=hw,mode='nearest')[:,0]

def valid_mask(gt):return torch.isfinite(gt)&(gt>.1)&(gt<9.99)

def metrics(pred,gt):
    p=np.asarray(pred,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    valid=np.isfinite(g)&(g>.1)&(g<9.99)
    if not np.isfinite(p).all() or valid.sum()<100:raise ValueError('Invalid prediction/reference')
    a=np.clip(p[valid],.1,10);b=g[valid];d=a-b
    return dict(abs_rel=float(np.mean(np.abs(d)/b)),rmse_m=float(np.sqrt(np.mean(d*d))),
                delta1=float(np.mean(np.maximum(a/b,b/a)<1.25)),valid_pixels=int(valid.sum()))

def validation(model,adapter,records,variant,stats):
    scores=[];adapter.eval()
    with torch.no_grad():
        for r in records:
            item=load_item(r);gt=np.load(ROOT/r['depth'])
            pred,_,_=predict_from_features(model,item['features'],item['patch_hw'],adapter,condition([item],variant,stats),gt.shape)
            scores.append(metrics(pred[0].numpy(),gt)['abs_rel'])
    adapter.train();return float(np.mean(scores))

def train_variant(model,records,variant,stats,args,config_hash):
    save=OUT/'models'/f'{variant}.pt';save.parent.mkdir(exist_ok=True)
    if save.exists():
        old=torch.load(save,map_location='cpu',weights_only=True)
        if old['configuration_sha256']!=config_hash:raise ValueError('Learned checkpoint configuration changed')
        adapter=ColorFeatureAdapter();adapter.load_state_dict(old['state_dict']);adapter.eval();return adapter,old
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    adapter=ColorFeatureAdapter();optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.0001)
    training=defaultdict(list)
    for r in records:
        if r['split']=='training':training[r['view_group']].append(r)
    validation_records=[r for r in records if r['split']=='validation']
    groups=sorted(training);rng=random.Random(SEED);history=[];best=float('inf');best_state=None;best_step=0;start=time.perf_counter()
    for step in range(1,args.steps+1):
        group=groups[rng.randrange(len(groups))];pair=rng.sample(training[group],2)
        items=[load_item(r) for r in pair];features=join_features(items);hw=items[0]['image_hw']
        gt=torch.cat([target(r,hw) for r in pair]);mask=valid_mask(gt)
        pred,adapted,reg=predict_from_features(model,features,items[0]['patch_hw'],adapter,condition(items,variant,stats))
        depth=(torch.abs(pred[mask]-gt[mask])/gt[mask]).mean()
        consistency=torch.zeros(())
        featloss=torch.zeros(())
        if variant.endswith('consistency'):
            both=mask[0]&mask[1]
            consistency=torch.abs(torch.log(pred[0][both].clamp_min(.1))-torch.log(pred[1][both].clamp_min(.1))).mean()
            featloss=invariant_feature_loss(adapted)
        loss=depth+.1*consistency+.05*featloss+.001*reg
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step()
        row=dict(step=step,loss=float(loss.detach()),depth_loss=float(depth.detach()),depth_consistency=float(consistency.detach()),feature_consistency=float(featloss.detach()))
        if step%50==0 or step==args.steps:
            val=validation(model,adapter,validation_records,variant,stats);row['validation_abs_rel']=val
            if val<best:
                best=val;best_step=step;best_state={k:v.detach().clone() for k,v in adapter.state_dict().items()}
            print(f'{variant}:step{step}/{args.steps},train{row["depth_loss"]:.4f},val{val:.4f},elapsed{time.perf_counter()-start:.0f}s',flush=True)
        history.append(row)
    adapter.load_state_dict(best_state);adapter.eval()
    record=dict(state_dict=best_state,configuration_sha256=config_hash,variant=variant,training_steps=args.steps,
                selected_step=best_step,validation_abs_rel=best,history=history,training_seconds=time.perf_counter()-start,
                trainable_parameters=sum(p.numel() for p in adapter.parameters()),descriptor_statistics=stats)
    torch.save(record,save)
    (OUT/f'{variant}_training.json').write_text(json.dumps({k:v for k,v in record.items() if k!='state_dict'},indent=2),encoding='utf-8')
    return adapter,record

def write_csv(rows,path):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def evaluate(model,adapter,records,method,stats):
    output=OUT/'predictions'/method;output.mkdir(parents=True,exist_ok=True);rows=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            path=output/f'{r["id"]}.npy'
            gt=np.load(ROOT/r['depth'])
            if path.exists():pred=np.load(path)
            else:
                item=load_item(r,'gray_world' if method=='gray_world_preprocess' else 'none')
                cond=None if adapter is None else condition([item],method,stats)
                p,_,_=predict_from_features(model,item['features'],item['patch_hw'],adapter,cond,gt.shape)
                pred=p[0].numpy().astype(np.float32);np.save(path,pred)
            rows.append(dict(id=r['id'],scene=r['scene'],view_group=r['view_group'],lighting=r['lighting'],method=method,**metrics(pred,gt)))
            if (i+1)%88==0:print(f'Evaluation {method}:{i+1}/{len(records)}',flush=True)
    return rows

def instability(records,method):
    groups=defaultdict(list)
    for r in records:groups[r['view_group']].append(r)
    output=[]
    for key,rs in sorted(groups.items()):
        gt=np.load(ROOT/rs[0]['depth']).astype(np.float64);valid=np.isfinite(gt)&(gt>.1)&(gt<9.99)
        ps=np.stack([np.clip(np.load(OUT/'predictions'/method/f'{r["id"]}.npy'),.1,10) for r in rs])
        score=float(np.mean(np.abs(ps[:,valid]-ps[:,valid].mean(axis=0))/gt[valid]))
        output.append(dict(view_group=key,scene=rs[0]['scene'],method=method,lighting_instability=score,n_lights=len(rs)))
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=ROOT/'phase2/data/vari/manifest.json')
    parser.add_argument('--input-size',type=int,default=224);parser.add_argument('--steps',type=int,default=150)
    parser.add_argument('--threads',type=int,default=8);parser.add_argument('--cache-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    payload,records=read_manifest(args.manifest);mhash=hashfile(args.manifest)
    config=dict(seed=SEED,input_size=args.input_size,steps=args.steps,threads=args.threads,manifest_sha256=mhash,
                base_checkpoint_sha256=hashfile(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'),
                encoder_frozen=True,decoder_frozen=True,variants=VARIANTS,learning_rate=.001,weight_decay=.0001,
                loss_weights={'depth':1.,'depth_consistency':.1,'feature_consistency':.05,'modulation':.001})
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()+(Path(__file__).parent/'cc_model.py').read_bytes()).hexdigest()
    config['configuration_sha256']=config_hash
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:raise ValueError('Run configuration differs; preserve old run and use a fresh results folder')
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    torch.manual_seed(SEED);torch.set_num_interop_threads(1);model=frozen_model(args.threads)
    start=time.perf_counter();make_cache(model,records,args.input_size,mhash)
    if args.cache_only:return
    stats=statistics(records);evaluation=[r for r in records if r['split']=='evaluation'];allrows=[];allinst=[];training=[]
    for method in EVAL_METHODS:
        adapter=None
        if method in VARIANTS:
            adapter,metadata=train_variant(model,records,method,stats,args,config_hash)
            training.append({k:v for k,v in metadata.items() if k not in ['state_dict','history']})
        rows=evaluate(model,adapter,evaluation,method,stats);allrows.extend(rows);allinst.extend(instability(evaluation,method))
        write_csv(allrows,OUT/'metrics.csv');write_csv(allinst,OUT/'instability.csv')
    results=[]
    for method in EVAL_METHODS:
        rows=[r for r in allrows if r['method']==method];irs=[r for r in allinst if r['method']==method]
        results.append(dict(method=method,n_images=len(rows),n_view_groups=len(irs),
            **{k:float(np.mean([r[k] for r in rows])) for k in ['abs_rel','rmse_m','delta1']},
            lighting_instability=float(np.mean([r['lighting_instability'] for r in irs]))))
    bylight=[]
    for method in EVAL_METHODS:
        for light in sorted({r['lighting'] for r in evaluation}):
            selected=[r for r in allrows if r['method']==method and r['lighting']==light]
            bylight.append(dict(method=method,lighting=light,n_images=len(selected),abs_rel=float(np.mean([r['abs_rel'] for r in selected])),rmse_m=float(np.mean([r['rmse_m'] for r in selected]))))
    summary=dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),configuration=config,dataset=payload,
        results=results,by_lighting=bylight,training=training,wall_seconds=time.perf_counter()-start,
        limitations=['Exploratory repartition of public Vari test release; official training release unavailable',
            'One evaluation room with multiple camera views, not multiple independent scenes',
            'Rendered lighting; real camera and real-light transfer remain unestablished',
            'CC-inspired internal feature adaptation does not demonstrate physical illuminant recovery or algorithm novelty',
            'Accuracy and lighting stability must be interpreted together; neither guarantees the other'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');write_csv(results,OUT/'method_summary.csv');write_csv(bylight,OUT/'lighting_summary.csv')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
