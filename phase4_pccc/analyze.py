"""Independent angular audit, complete tables and non-selected visual evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upstream import ROOT,sha,identities,load,make_model,predict

HOME=ROOT/'phase4_pccc'
NAMES={'released_real':'Released sum / real geometry','released_flat':'Released sum / flat geometry',
       'weighted_real':'Restored weighted / real geometry','weighted_flat':'Restored weighted / flat geometry',
       'gray_world_points':'Gray-World / sampled RGB','training_mean_illuminant':'Training mean illuminant'}
PALETTE=['#007D79','#9BBEB8','#23485F','#8EACC0','#B08A5A','#8E819C']

def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def independent_angle(p,y):
    # atan2 is independent of the training/evaluation acos implementation.
    a=np.asarray(p,dtype=np.float64);b=np.asarray(y,dtype=np.float64)
    assert np.isfinite(a).all() and np.linalg.norm(a)>0 and np.linalg.norm(b)>0
    angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))
    floor=np.degrees(np.arccos(.999999))
    return float(np.clip(angle,floor,180-floor))

def describe(errors):
    e=np.sort(np.asarray(errors,dtype=np.float64));n=len(e)
    return {'n':n,'mean_deg':float(e.mean()),'median_deg':float(np.median(e)),
        'trimean_deg':float((np.percentile(e,25)+2*np.percentile(e,50)+np.percentile(e,75))/4),
        'best25_deg':float(e[:int(.25*n)].mean()),'worst25_deg':float(e[int(.75*n):].mean()),
        'percentile95_deg':float(np.percentile(e,95))}

def flat_input(x):
    x=x.copy();v=np.abs(x[:,2])>1e-8;x[v,:3]/=np.abs(x[v,2:3]);return x

def audit(out,cfg,manifest,rows):
    identity=dict(cfg);expected=identity.pop('sha256')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==expected
    assert cfg['upstream']==identities() and sha(HOME/'manifest.json')==cfg['manifest_sha256']
    for name,value in cfg['sources'].items():assert sha(HOME/name)==value
    records={r['id']:r for r in manifest['samples']};test=sorted([r['id'] for r in manifest['samples'] if r['split']=='test'],key=int)
    expected_keys={(key,m,str(s)) for key in test for m in cfg['variants'] for s in cfg['seeds']}
    expected_keys|={(key,m,'none') for key in test for m in ['gray_world_points','training_mean_illuminant']}
    assert {(r['id'],r['method'],r['seed']) for r in rows}==expected_keys and len(rows)==len(expected_keys)==868
    for record in records.values():
        for field in ['cloud','label']:assert sha(ROOT/record[field])==record[field+'_sha256']
    maximum=0.;by_key={}
    for r in rows:
        y=np.load(ROOT/records[r['id']]['label']).astype(np.float32)
        p=np.array([float(r['pred_'+c]) for c in ['r','g','b']])
        assert np.array_equal(y.astype(np.float64),[float(r['gt_'+c]) for c in ['r','g','b']])
        error=independent_angle(p,y);difference=abs(error-float(r['angular_error_deg']))
        maximum=max(maximum,difference);assert difference<1e-7
        by_key[(r['id'],r['method'],r['seed'])]=p
    author,_,_=load();fingerprint=json.loads((out/'evaluation_sources.json').read_text())
    checkpoints=[];prediction_difference=0.
    for method,(head,geometry,sigma) in cfg['variants'].items():
        for seed in cfg['seeds']:
            path=out/'models'/f'{method}_{seed}.pt'
            assert sha(path)==fingerprint['checkpoint_sha256'][path.name]
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert checkpoint['configuration_sha256']==cfg['sha256']
            assert len(checkpoint['history'])==cfg['epochs']
            best=min(checkpoint['history'],key=lambda r:r['validation_mean_deg'])
            assert best['epoch']==checkpoint['selected_epoch'] and best['validation_mean_deg']==checkpoint['validation_mean_deg']
            model=make_model(author,head);model.load_state_dict(checkpoint['state_dict']);model.eval()
            assert sum(p.numel() for p in model.parameters())==checkpoint['trainable_parameters']
            with torch.inference_mode():
                for start in range(0,len(test),16):
                    ids=test[start:start+16];inputs=[]
                    for key in ids:
                        x=np.load(ROOT/records[key]['cloud']).astype(np.float32)
                        if geometry=='flat':x=flat_input(x)
                        inputs.append(x.T.copy())
                    p=predict(model,torch.from_numpy(np.stack(inputs)),head).numpy()
                    for key,value in zip(ids,p):
                        d=float(np.max(np.abs(value-by_key[(key,method,str(seed))])))
                        prediction_difference=max(prediction_difference,d);assert d<2e-5
            checkpoints.append({'method':method,'seed':seed,'selected_epoch':best['epoch'],
                'validation_mean_deg':best['validation_mean_deg'],'checkpoint_sha256':sha(path),
                'trainable_parameters':checkpoint['trainable_parameters'],'training_seconds':checkpoint['training_seconds']})
    # Independently verify both non-neural baselines, including training-only labels.
    train_y=np.stack([np.load(ROOT/r['label']).astype(np.float32) for r in records.values() if r['split']=='train'])
    mean=(train_y/np.linalg.norm(train_y,axis=1,keepdims=True)).mean(0)
    for key in test:
        x=np.load(ROOT/records[key]['cloud']).astype(np.float32)
        assert np.array_equal(x[:,3:].mean(0).astype(np.float64),by_key[(key,'gray_world_points','none')])
        assert np.allclose(mean,by_key[(key,'training_mean_illuminant','none')],atol=1e-7)
    result={'audit':'PASS','test_samples':62,'prediction_rows':len(rows),'trained_models_checked':len(checkpoints),
        'independent_atan2_max_error_deg':maximum,'max_reinference_prediction_component_difference':prediction_difference,
        'all_frozen_inputs_sources_and_checkpoints_match':True,'validation_selection_verified':True,
        'test_used_for_checkpoint_selection':False,'baselines_recomputed':True,
        'predictions_csv_sha256':sha(out/'predictions.csv'),'checkpoints':checkpoints}
    dump(out/'independent_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'},indent=2))
    return result

def summarize(out,cfg,rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['method'],r['seed'])].append(float(r['angular_error_deg']))
    per_run=[{'method':m,'seed':s,**describe(e)} for (m,s),e in grouped.items()]
    writecsv(out/'per_seed_metrics.csv',per_run)
    table=[]
    for method in NAMES:
        items=[r for r in per_run if r['method']==method]
        row={'method':method,'training_seeds':len(items),'test_captures':62}
        for metric in ['mean_deg','median_deg','trimean_deg','best25_deg','worst25_deg','percentile95_deg']:
            values=[r[metric] for r in items];row[metric]=float(np.mean(values));row[metric+'_seed_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        table.append(row)
    writecsv(out/'summary.csv',table)
    effects={}
    for family in ['released','weighted']:
        paired=[];per_seed=[]
        for seed in cfg['seeds']:
            a={r['id']:float(r['angular_error_deg']) for r in rows if r['method']==family+'_real' and r['seed']==str(seed)}
            b={r['id']:float(r['angular_error_deg']) for r in rows if r['method']==family+'_flat' and r['seed']==str(seed)}
            keys=sorted(a,key=int);difference=np.array([a[k]-b[k] for k in keys]);paired.append(difference)
            per_seed.append({'seed':seed,'real_minus_flat_mean_deg':float(difference.mean())})
        difference=np.mean(paired,axis=0);rng=np.random.default_rng(20261109)
        draw=rng.integers(0,len(difference),size=(5000,len(difference)))
        ci=np.quantile(difference[draw].mean(1),[.025,.975])
        effects[family]={'mean_real_minus_flat_deg':float(difference.mean()),
            'conditional_capture_bootstrap_95_interval':ci.tolist(),'per_seed':per_seed,
            'test_captures':len(difference),'real_better_captures_after_seed_average':int((difference<0).sum()),
            'note':'Negative favors geometry. Resamples62captures, conditional on3trainedseeds; not proven room-independent population inference.'}
    dump(out/'geometry_effects.json',effects);return table,effects

def figures(out,cfg,manifest,rows,table):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,5))
    values=[r['mean_deg'] for r in table];std=[r['mean_deg_seed_std'] for r in table]
    ax.barh(range(len(table)),values,xerr=std,color=PALETTE,capsize=4)
    ax.set_yticks(range(len(table)),[NAMES[r['method']] for r in table]);ax.invert_yaxis()
    ax.set_xlabel('Mean recovery angular error (degrees); lower is better')
    ax.set_title('Real DepthAWB / official fold 0 / 62 held-out captures',fontweight='bold')
    for i,v in enumerate(values):ax.text(v+std[i]+.03,i,f'{v:.2f}',va='center')
    ax.set_xlim(0,max(v+s for v,s in zip(values,std))*1.18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.015,'Learned methods: mean +/- training-seed SD (3 seeds). Reduced-budget pilot, not the published benchmark.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(folder/'angular_error_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,(method,mode),color in zip(axes.ravel(),cfg['variants'].items(),PALETTE):
        for seed in cfg['seeds']:
            h=json.loads((out/'histories'/f'{method}_{seed}.json').read_text())
            ax.plot([r['epoch'] for r in h['history']],[r['validation_mean_deg'] for r in h['history']],label=str(seed),alpha=.8)
            ax.scatter([h['selected_epoch']],[h['validation_mean_deg']],s=22)
        ax.set_title(NAMES[method]);ax.set_xlabel('Epoch');ax.set_ylabel('Inner-validation error (deg)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Checkpoint selection uses inner validation only',fontweight='bold');fig.tight_layout()
    fig.savefig(folder/'validation_curves.png',dpi=160);plt.close(fig)
    # No ranking on prediction errors: deterministic first numeric test ID.
    r=min([r for r in manifest['samples'] if r['split']=='test'],key=lambda r:int(r['id']))
    x=np.load(ROOT/r['cloud']);gt=np.load(ROOT/r['label']);valid=np.abs(x[:,2])>1e-8
    fig=plt.figure(figsize=(11,6));ax=fig.add_subplot(2,3,1,projection='3d')
    ax.scatter(*x[valid,:3].T,c=np.clip(x[valid,3:],0,1)**(1/2.2),s=12)
    ax.set_title('Native colored point cloud\nzero-geometry points omitted');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
    ax=fig.add_subplot(2,3,2);ax.imshow(np.clip(x[:,3:].reshape(16,16,3),0,1)**(1/2.2));ax.axis('off');ax.set_title('Released 16x16 RGB grid\ndisplay gamma only')
    ax=fig.add_subplot(2,3,3);ax.imshow(np.tile((gt/gt.max())[None,None],(16,16,1)));ax.axis('off');ax.set_title('Reference illuminant\nnormalized display swatch')
    for i,method in enumerate(['released_real','weighted_real','weighted_flat'],4):
        selected=next(t for t in rows if t['id']==r['id'] and t['method']==method and t['seed']==str(cfg['seeds'][0]))
        p=np.array([float(selected['pred_'+c]) for c in ['r','g','b']]);gain=p/p[1]
        corrected=np.clip(x[:,3:]/gain,0,1)**(1/2.2)
        ax=fig.add_subplot(2,3,i);ax.imshow(corrected.reshape(16,16,3));ax.axis('off')
        ax.set_title(NAMES[method]+f"\nerror {float(selected['angular_error_deg']):.2f} deg",fontsize=9)
    fig.suptitle(f'DepthAWB sample {r["id"]}: first numeric test ID; fixed first training seed',fontweight='bold')
    fig.text(.5,.015,'Thumbnails derive from the authors\' sampled clouds; they are not original full-resolution photographs or a local-AWB evaluation.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(folder/'real_example.png',dpi=170);plt.close(fig)
    dump(folder/'selection.json',{'id':r['id'],'seed':cfg['seeds'][0],'rule':'First numeric held-out ID and first predeclared seed; no error ranking.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HOME/'results/cpu_v1');args=parser.parse_args()
    out=args.out;cfg=json.loads((out/'config.json').read_text());manifest=json.loads((HOME/'manifest.json').read_text())
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    assert (out/'evaluation_complete.json').exists(),'Refuse partial test results'
    rows=list(csv.DictReader((out/'predictions.csv').open(newline='',encoding='utf-8')))
    result=audit(out,cfg,manifest,rows);table,effects=summarize(out,cfg,rows);figures(out,cfg,manifest,rows,table)
    dump(out/'summary.json',{'scope':'Adapted small CPU reproduction; not full paper benchmark',
        'table':table,'geometry_effects':effects,'audit':'PASS','test_samples':62,
        'all_methods_and_seeds_reported':True,'total_training_seconds':sum(r['training_seconds'] for r in result['checkpoints']),
        'published_accuracy_reproduced':False,'new_algorithm_novelty_claimed':False})
    print(json.dumps({'analysis':'COMPLETE','models':12,'test_predictions':len(rows),'summary':str(out/'summary.json')},indent=2))

if __name__=='__main__':main()
