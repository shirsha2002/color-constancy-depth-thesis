"""Run a resumable paired color-constancy / metric-depth pilot on real data."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/Depth-Anything-V2/metric_depth'))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from science import CASTS,METHODS,evaluate_depth,transform

def load_records(path):
    payload=json.loads(path.read_text(encoding='utf-8'))
    records=payload if isinstance(payload,list) else payload.get('images',payload.get('records',payload.get('samples',[])))
    if not records:raise ValueError('Manifest contains no image records')
    return payload,records

def read_image(record):
    import cv2
    rgb_path=ROOT/record.get('rgb',record.get('rgb_path',''))
    depth_path=ROOT/record.get('depth',record.get('depth_path',''))
    image=cv2.imread(str(rgb_path))
    if image is None:raise FileNotFoundError(rgb_path)
    gt=np.load(depth_path).astype(np.float32)
    if image.shape[:2]!=gt.shape:raise ValueError('Unaligned RGB/depth')
    return image[:,:,::-1].astype(np.float32)/255,gt

def create_model(threads):
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    checkpoint=ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    return model

def git_revision(path):
    try:return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def write_csv(rows,path):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate(rows):
    result=[]
    for condition in CASTS:
        for method in METHODS:
            selected=[r for r in rows if r['condition']==condition and r['method']==method]
            if not selected:continue
            rec=dict(condition=condition,method=method,n_images=len(selected))
            for metric in ['abs_rel','rmse_m','mae_m','delta1','inference_seconds','cast_clipped_fraction','correction_clipped_fraction']:
                rec[metric]=float(np.mean([r[metric] for r in selected]))
            result.append(rec)
    return result

def paired_bootstrap(rows,method,conditions,seed=20261023):
    """Resample images, retaining all casts for each sampled image."""
    ids=sorted({r['image_id'] for r in rows})
    differences=[]
    for image_id in ids:
        selected=[r for r in rows if r['image_id']==image_id and r['condition'] in conditions]
        a=[r['abs_rel'] for r in selected if r['method']=='none']
        b=[r['abs_rel'] for r in selected if r['method']==method]
        if len(a)==len(conditions) and len(b)==len(conditions):differences.append(float(np.mean(a)-np.mean(b)))
    if not differences:return None
    d=np.asarray(differences);rng=np.random.default_rng(seed)
    boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
    return dict(method=method,metric='abs_rel_reduction',mean=float(d.mean()),ci95_low=float(np.percentile(boot,2.5)),
                ci95_high=float(np.percentile(boot,97.5)),n_images=len(d),improved_images=int((d>0).sum()),
                harmed_images=int((d<0).sum()),bootstrap_unit='image',resamples=10000)

def summarize(rows,manifest,records,args,elapsed):
    evaluated=[r for r in rows if r['split'] in ('evaluation','test','val','validation')]
    if not evaluated:evaluated=rows
    agg=aggregate(evaluated)
    corrupt=[c for c in CASTS if c!='clean']
    method_summaries=[]
    for method in METHODS:
        rr=[r for r in evaluated if r['condition']!='clean' and r['method']==method]
        clean=[r for r in evaluated if r['condition']=='clean' and r['method']==method]
        method_summaries.append(dict(method=method,n_images=len({r['image_id'] for r in rr}),
            corrupted_abs_rel=float(np.mean([r['abs_rel'] for r in rr])),
            corrupted_rmse_m=float(np.mean([r['rmse_m'] for r in rr])),
            corrupted_delta1=float(np.mean([r['delta1'] for r in rr])),
            clean_abs_rel=float(np.mean([r['abs_rel'] for r in clean])),
            clean_rmse_m=float(np.mean([r['rmse_m'] for r in clean])),
            clean_delta1=float(np.mean([r['delta1'] for r in clean]))))
    checkpoint=ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    payload=dict(title='Color constancy and monocular depth: first controlled pilot',
        thesis_title='Design and Implementation of a Color-Constancy-Assisted Monocular Depth Estimation System',
        student={'name':'Shirsha Mitra','id':'2023521460130'},
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        dataset=manifest,
        protocol=dict(model='Depth Anything V2 Metric Hypersim Small',encoder='vits',parameters='24.8M',
            upstream_git_revision=git_revision(ROOT/'external/Depth-Anything-V2'),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            input_size=args.input_size,model_max_depth_m=20,evaluation_min_depth_m=.1,evaluation_max_depth_m=10,
            crop={'at_480x640':[45,471,41,601],'description':'rows45:471,columns41:601; scaled proportionally for other dimensions'},
            prediction_clipping_m=[.1,10],scale_alignment=False,
            aggregation='Arithmetic mean of per-image metrics; six casts equally weighted',
            casts={k:list(v) for k,v in CASTS.items()},methods=METHODS,
            transform='sRGB to linear RGB; diagonal channel gains; preserve mean luminance before clipping; clip; return to sRGB',
            half_strength='Square root of Gray-World gains (log-gain strength0.5); fixed before evaluation',
            quantization='Model receives RGB-to-BGR uint8 images after transformation',
            selection='Fixed methods and gains; no method parameters fitted to evaluation ground truth'),
        runtime=dict(device='CPU',cpu='Intel Core Ultra 7 255H',ram_gb=32,threads=args.threads,wall_seconds=elapsed,
            unique_predictions=len(rows),evaluation_images=len({r['image_id'] for r in evaluated}),
            development_images=len({r['image_id'] for r in rows if r['split'] not in ('evaluation','test','val','validation')}),
            mean_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows])),
            total_recorded_inference_seconds=float(np.sum([r['inference_seconds'] for r in rows])),platform=platform.platform()),
        results=method_summaries,by_condition=agg,
        bootstrap=[paired_bootstrap(evaluated,m,corrupt) for m in METHODS if m!='none'],
        limitations=['Small deterministic pilot subset, not a full NYU benchmark reproduction',
            'Synthetic global color casts do not reproduce all real illumination, camera processing, or spatially mixed lighting',
            'Reference depth includes dataset preprocessing and sensor limitations',
            'One pretrained model, no network training and no established novel algorithm contribution',
            'Per-image bootstrap does not establish generalization to all scenes or datasets'])
    return payload

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=Path('data/nyu_pilot/manifest.json'))
    parser.add_argument('--input-size',type=int,default=518)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--split',default='all')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    manifest,records=load_records(args.manifest)
    if args.split!='all':records=[r for r in records if r['split']==args.split]
    if args.limit:records=records[:args.limit]
    import torch
    model=create_model(args.threads)
    rgb,_=read_image(records[0]);bgr=np.round(rgb[:,:,::-1]*255).astype(np.uint8)
    with torch.inference_mode():
        model.infer_image(bgr,input_size=args.input_size)
        if args.benchmark:
            times=[]
            for i in range(3):
                start=time.perf_counter();p=model.infer_image(bgr,input_size=args.input_size);times.append(time.perf_counter()-start)
            print(json.dumps(dict(threads=args.threads,input_size=args.input_size,seconds=times,
                                  shape=list(p.shape),depth_range=[float(p.min()),float(p.max())]),indent=2));return
    out=ROOT/'results';cache=out/'predictions';cache.mkdir(parents=True,exist_ok=True)
    fingerprint_data=args.manifest.read_bytes()+(ROOT/'scripts/science.py').read_bytes()+str(args.input_size).encode()
    fingerprint_data+=hashlib.sha256((ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth').read_bytes()).digest()
    fingerprint=hashlib.sha256(fingerprint_data).hexdigest()
    config=dict(input_size=args.input_size,threads=args.threads,manifest=str(args.manifest),configuration_sha256=fingerprint)
    (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[];total=len(records)*len(CASTS)*len(METHODS);start=time.perf_counter()
    with torch.inference_mode():
        for record in records:
            image_id=str(record.get('id',record.get('image_id')));rgb,gt=read_image(record)
            for condition in CASTS:
                for method in METHODS:
                    transformed,meta=transform(rgb,condition,method)
                    stem=f'{image_id}__{condition}__{method}';predpath=cache/f'{stem}.npy';metapath=cache/f'{stem}.json'
                    if predpath.exists() and metapath.exists():
                        saved=json.loads(metapath.read_text());
                        if saved.get('input_size')!=args.input_size or saved.get('threads')!=args.threads or saved.get('configuration_sha256')!=fingerprint:
                            raise ValueError('Cached configuration differs; use another results directory')
                        pred=np.load(predpath);seconds=saved['seconds']
                    else:
                        bgr=np.round(np.clip(transformed[:,:,::-1],0,1)*255).astype(np.uint8)
                        tick=time.perf_counter();pred=model.infer_image(bgr,input_size=args.input_size);seconds=time.perf_counter()-tick
                        np.save(predpath,pred.astype(np.float32))
                        metapath.write_text(json.dumps(dict(seconds=seconds,input_size=args.input_size,threads=args.threads,configuration_sha256=fingerprint)),encoding='utf-8')
                    metrics=evaluate_depth(pred,gt)
                    row=dict(image_id=image_id,split=record['split'],scene=record.get('scene','unknown'),condition=condition,method=method,
                             **metrics,inference_seconds=seconds,cast_clipped_fraction=meta['cast_clipped_fraction'],
                             correction_clipped_fraction=meta['correction_clipped_fraction'])
                    rows.append(row)
                    if len(rows)%8==0 or len(rows)==total:
                        print(f'{len(rows)}/{total} predictions; {image_id} {condition} {method}; elapsed{time.perf_counter()-start:.0f}s',flush=True)
            write_csv(rows,out/'metrics.csv')
    elapsed=time.perf_counter()-start
    write_csv(rows,out/'metrics.csv')
    summary=summarize(rows,manifest,records,args,elapsed)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(summary['by_condition'],out/'condition_summary.csv')
    write_csv(summary['results'],out/'method_summary.csv')
    print('COMPLETE',json.dumps(summary['results'],indent=2),flush=True)

if __name__=='__main__':main()
