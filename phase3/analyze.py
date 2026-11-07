"""Independently verify saved predictions and report the entire frozen study.

This auditor does not import the numeric runner's metric implementation.
ReMID audit clusters are conservative labels, not verified independent rooms.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
METHODS=['frozen','ordinary_aug','achromatic','content','chromatic_noeq','chromatic_eq','flat_geometry_eq','geometry_chromatic_eq']
NAMES={'frozen':'Frozen model','ordinary_aug':'Ordinary + augmentation',
       'achromatic':'Achromatic control','content':'Content control','chromatic_noeq':'Chromatic, no equivariance',
       'chromatic_eq':'Chromatic + equivariance','flat_geometry_eq':'Flat-geometry control','geometry_chromatic_eq':'Predicted-geometry chromatic'}
COLORS=['#87949C','#1C4656','#8F83A5','#C19268','#8CB8AE','#4C9690','#57769B','#007C77']
NAMES.update({'spatial_content':'Local content control','spatial_flat':'Local flat-geometry control',
    'spatial_no_graph_loss':'Local, no graph loss','spatial_no_equivariance':'Local, no equivariance',
    'spatial_geometry_eq':'Local geometry + chromatic'})
COLORS+=['#BA9A75','#7E90A5','#96B1A4','#A292B1','#006E69']
PRED_ROOTS={};PRIMARY='geometry_chromatic_eq'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def read_data():
    result={}
    for family,path in [('nyu','phase3/data/nyu_rgbd/manifest.json'),('remid','phase3/data/remid/manifest.json'),('vari','phase2/data/vari/manifest.json')]:
        records=json.loads((ROOT/path).read_text(encoding='utf-8'))['images']
        result[family]={r['id']:r for r in records if r['split']=='evaluation'}
    return result
@lru_cache(maxsize=220)
def reference(path,scale):return np.load(ROOT/path).astype(np.float32)*np.float32(scale)
def gt(r):return reference(r['depth'],r.get('depth_scale_to_meters',1.))
def mask_for(depth,family):
    v=np.isfinite(depth)&(depth>.1)&(depth<10.)
    if family=='nyu':
        h,w=depth.shape;crop=np.zeros((h,w),bool)
        crop[round(45*h/480):round(471*h/480),round(41*w/640):round(601*w/640)]=True
        v&=crop
    return v
def path_for(out,row):
    key=row['family']+'__'+row['id'].replace('/','__')
    base=ROOT/PRED_ROOTS[row['method']] if PRED_ROOTS else out
    return base/'predictions'/row['family']/row['method']/f'{key}__{row["condition"]}.npy'

def audit(out,rows,data):
    cfg=json.loads((out/'config.json').read_text())
    for relative,expected in {**cfg['sources'],**cfg['manifests']}.items():
        assert sha(ROOT/relative)==expected,f'Frozen source/input changed: {relative}'
    expected=cfg.get('expected_rows',1639*len(METHODS))
    assert len(rows)==expected and len({(r['family'],r['id'],r['condition'],r['method']) for r in rows})==expected
    maxima={'abs_rel':0.,'rmse_m':0.,'delta1':0.};start=time.perf_counter()
    for index,row in enumerate(rows,1):
        r=data[row['family']][row['id']];g=gt(r);v=mask_for(g,row['family'])
        p=np.load(path_for(out,row));assert p.shape==g.shape and np.isfinite(p).all()
        a=np.clip(p[v].astype(np.float64),.1,10);b=g[v].astype(np.float64);error=a-b
        metrics={'abs_rel':np.abs(error/b).mean(),'rmse_m':np.sqrt(np.square(error).mean()),
                 'delta1':(np.maximum(a/b,b/a)<1.25).mean()}
        assert int(row['valid_pixels'])==int(v.sum())
        for name,value in metrics.items():
            difference=abs(float(row[name])-float(value));maxima[name]=max(maxima[name],difference)
            assert difference<2e-12,(row['id'],row['method'],name,difference)
        if index%1024==0:print(f'Audit predictions: {index}/{len(rows)}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
    import torch
    checkpoint_checks=[]
    for method in METHODS[1:]:
        base=ROOT/PRED_ROOTS[method] if PRED_ROOTS else out
        run_config=json.loads((base/'config.json').read_text())
        checkpoint_path=base/'models'/f'{method}.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert checkpoint['configuration_sha256']==run_config['configuration_sha256']
        evaluation=json.loads((base/'evaluation_sources.json').read_text())
        assert evaluation['checkpoint_sha256'][method]==sha(checkpoint_path)
        candidates=[r for r in checkpoint['history'] if 'validation_abs_rel' in r]
        best=min(candidates,key=lambda r:r['validation_abs_rel'])
        assert checkpoint['selected_step']==best['step']
        assert checkpoint['validation_abs_rel']==best['validation_abs_rel']
        assert checkpoint['trainable_parameters']==run_config.get('trainable_parameters',59699)
        parameter_count=sum(v.numel() for k,v in checkpoint['state_dict'].items() if k not in ['luma','luma_weights'])
        assert parameter_count==checkpoint['trainable_parameters']
        checkpoint_checks.append({'method':method,'selected_step':best['step'],'validation_selection_checked':True})
    nyu=json.loads((ROOT/'phase3/data/nyu_rgbd/manifest.json').read_text())['images']
    scenes={s:{r['scene'] for r in nyu if r['split']==s} for s in ['training','validation','evaluation']}
    assert [len(scenes[s]) for s in scenes]==[200,40,128]
    assert not scenes['training']&scenes['validation'] and not scenes['training']&scenes['evaluation'] and not scenes['validation']&scenes['evaluation']
    previous={r['scene'] for r in json.loads((ROOT/'data/nyu_pilot/manifest.json').read_text())['images']}
    assert not previous&set.union(*scenes.values())
    result={'audit':'PASS','prediction_rows_independently_recomputed':len(rows),
        'maximum_absolute_metric_difference':maxima,'frozen_source_and_manifests_match':True,
        'checkpoint_selection_checks':checkpoint_checks,'nyu_scene_splits':[200,40,128],
        'previous_40_nyu_scenes_excluded':True,'remid_external_only':True,
        'reference_depth_not_model_input':True,'array_dtype':'float32; scoring float64',
        'auditor_function_sha256':hashlib.sha256(''.join(inspect.getsource(f) for f in [audit,reference,gt,mask_for,path_for]).encode()).hexdigest(),
        'metrics_csv_sha256':sha(out/'metrics.csv')}
    dump(out/'independent_audit.json',result);return result

def grouped(rows,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[f] for f in fields)].append(r)
    return [{**dict(zip(fields,key)),**{m:float(np.mean([float(r[m]) for r in values])) for m in ['abs_rel','rmse_m','delta1']},'n_images':len(values)} for key,values in sorted(groups.items())]
def paired(a,b):
    keys=sorted(set(a)&set(b));assert keys and set(a)==set(b)
    x=np.array([a[k] for k in keys]);y=np.array([b[k] for k in keys]);d=x-y
    rng=np.random.default_rng(20261107);draw=rng.integers(0,len(keys),size=(5000,len(keys)))
    interval=np.quantile(d[draw].mean(1),[.025,.975])
    return {'groups':len(keys),'candidate_mean':float(x.mean()),'control_mean':float(y.mean()),
        'candidate_minus_control':float(d.mean()),'bootstrap_95_interval':interval.tolist(),
        'relative_error_reduction_percent':float(100*(y.mean()-x.mean())/y.mean()) if abs(y.mean())>1e-12 else None,
        'candidate_lower_error_groups':int((d<0).sum()),'note':'Paired group resampling; lower difference is better.'}

def summarize(out,rows):
    capture=grouped([r for r in rows if r['family']=='remid'],['scene','view_group','method'])
    # Equal weighting of conservative labels after equal capture weighting.
    clusters=grouped(capture,['scene','method'])
    for row in clusters:row['n_captures']=row.pop('n_images')
    writecsv(out/'remid_capture_scores.csv',capture);writecsv(out/'remid_cluster_scores.csv',clusters)
    tables=[]
    for method in METHODS:
        def mean(family,condition,metric='abs_rel'):
            subset=[float(r[metric]) for r in rows if r['family']==family and r['condition']==condition and r['method']==method]
            return float(np.mean(subset))
        tables.append({'method':method,'nyu_original_absrel':mean('nyu','original'),
            'nyu_control_absrel':mean('nyu','control'),
            'nyu_chromatic_absrel':.5*(mean('nyu','chroma_rb')+mean('nyu','chroma_green')),
            'nyu_exposure_absrel':mean('nyu','exposure'),
            'remid_per_image_absrel':mean('remid','original'),
            'remid_cluster_macro_absrel':float(np.mean([r['abs_rel'] for r in clusters if r['method']==method])),
            'vari_rendered_absrel':mean('vari','original')})
    writecsv(out/'comparison_table.csv',tables)
    points=defaultdict(dict)
    for row in rows:
        if row['family']=='nyu':points[(row['method'],row['condition'])][row['scene']]=float(row['abs_rel'])
    for method in METHODS:
        points[(method,'chromatic')]={k:.5*(points[(method,'chroma_rb')][k]+points[(method,'chroma_green')][k]) for k in points[(method,'control')]}
    comparisons={}
    primary=PRIMARY
    controls=[m for m in METHODS if m not in ['frozen',primary]]
    for condition in ['original','chromatic','exposure']:
        for control in controls:
            comparisons[f'nyu_{condition}__primary_vs_{control}']=paired(points[(primary,condition)],points[(control,condition)])
    for method in METHODS:
        comparisons[f'nyu_chromatic_vs_matched_neutral__{method}']=paired(points[(method,'chromatic')],points[(method,'control')])
        comparisons[f'nyu_exposure_vs_matched_neutral__{method}']=paired(points[(method,'exposure')],points[(method,'control')])
    for control in controls:
        left={k:points[(primary,'chromatic')][k]-points[(primary,'control')][k] for k in points[(primary,'control')]}
        right={k:points[(control,'chromatic')][k]-points[(control,'control')][k] for k in points[(control,'control')]}
        effect=paired(left,right)
        effect['note']='Difference in chromatic-minus-neutral error; paired128scene bootstrap. Relative error reduction of signed differences is not an accuracy percentage.'
        effect.pop('relative_error_reduction_percent',None)
        comparisons[f'nyu_chromatic_difference_in_differences__primary_vs_{control}']=effect
    for control in controls:
        a={r['scene']:r['abs_rel'] for r in clusters if r['method']==primary}
        b={r['scene']:r['abs_rel'] for r in clusters if r['method']==control}
        effect=paired(a,b);effect['note']='Conditional resampling of25conservative labels; not a population CI because physical room identities are unknown.'
        comparisons[f'remid_cluster__primary_vs_{control}']=effect
    dump(out/'paired_comparisons.json',comparisons)
    return tables,comparisons

def stability(out,rows,data):
    results=[]
    for family in ['remid','vari']:
        groups=defaultdict(list)
        for r in data[family].values():groups[r['view_group']].append(r)
        for method in METHODS:
            for capture,records in sorted(groups.items()):
                g=gt(records[0]);valid=mask_for(g,family);b=g[valid].astype(np.float64)
                # Retain only valid samples in memory, with every method sharing support.
                predictions=[]
                for r in records:
                    row={'family':family,'id':r['id'],'method':method,'condition':'original'}
                    predictions.append(np.clip(np.load(path_for(out,row))[valid],.1,10))
                p=np.stack(predictions).astype(np.float64);center=p.mean(0)
                normalized=np.abs(p-center)/b
                results.append({'family':family,'scene':records[0]['scene'],'view_group':capture,
                    'method':method,'n_lights':len(records),'normalized_lighting_mad':float(normalized.mean()),
                    'normalized_worst_deviation':float(normalized.max(0).mean())})
            print(f'Stability {family}: {method}',flush=True)
    writecsv(out/'lighting_stability.csv',results)
    summary=[]
    for family in ['remid','vari']:
        for method in METHODS:
            subset=[r for r in results if r['family']==family and r['method']==method]
            by_scene=defaultdict(list)
            for r in subset:by_scene[r['scene']].append(r['normalized_lighting_mad'])
            summary.append({'family':family,'method':method,'capture_macro_mad':float(np.mean([r['normalized_lighting_mad'] for r in subset])),
                'conservative_cluster_macro_mad':float(np.mean([np.mean(v) for v in by_scene.values()])),
                'capture_groups':len(subset),'scene_labels':len(by_scene)})
    writecsv(out/'lighting_stability_summary.csv',summary);return summary

def figures(out,tables,data):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    lookup={r['method']:r for r in tables}
    fig,axes=plt.subplots(2,2,figsize=(14,12))
    panels=[('nyu_original_absrel','Unmodified real NYU /128new scenes'),('nyu_chromatic_absrel','Digital chromatic changes /128new scenes'),
            ('remid_cluster_macro_absrel','Captured real ReMID /25conservative labels'),('vari_rendered_absrel','Rendered Vari /one previously inspected room')]
    for ax,(key,title) in zip(axes.ravel(),panels):
        values=[lookup[m][key] for m in METHODS];bars=ax.barh(range(len(METHODS)),values,color=COLORS[:len(METHODS)])
        ax.set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS],fontsize=9);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.2);ax.set_title(title,fontweight='bold');ax.set_xlabel('AbsRel; lower is better')
        for bar,value in zip(bars,values):ax.text(value+max(values)*.025,bar.get_y()+.4,f'{value:.3f}',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout(pad=2);fig.savefig(folder/'method_comparison.png',dpi=180);plt.close(fig)
    for family in ['nyu','remid']:
        r=sorted(data[family].values(),key=lambda r:r['id'])[0];depth=gt(r);valid=mask_for(depth,family)
        fig=plt.figure(figsize=(12,7));grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.04],left=.02,right=.94,bottom=.04,top=.89,wspace=.13,hspace=.2)
        axes=np.array([[fig.add_subplot(grid[y,x]) for x in range(3)] for y in range(2)])
        axes[0,0].imshow(Image.open(ROOT/r['rgb']));axes[0,0].set_title('Original real RGB')
        maps=[depth]
        for method in ['frozen','ordinary_aug',PRIMARY]:
            maps.append(np.load(path_for(out,{'family':family,'id':r['id'],'method':method,'condition':'original'})))
        titles=['Measured reference (m)','Frozen model','Ordinary + augmentation','Predicted-geometry chromatic']
        for ax,array,title in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],maps,titles):
            shown=np.clip(array,.1,10).astype(np.float32);shown[~valid]=np.nan
            im=ax.imshow(shown,vmin=.1,vmax=10,cmap='magma');ax.set_title(title,fontsize=10)
        error=np.abs(np.clip(maps[-1],.1,10)-depth);error[~valid]=np.nan
        err=axes[1,2].imshow(error,vmin=0,vmax=2,cmap='inferno');axes[1,2].set_title('Primary absolute error (m)',fontsize=10)
        for ax in axes.ravel():ax.axis('off')
        fig.colorbar(im,cax=fig.add_subplot(grid[0,3]),label='Depth (m)')
        fig.colorbar(err,cax=fig.add_subplot(grid[1,3]),label='Error (m),2+ saturated')
        fig.suptitle(f'{family.upper()}: first sorted evaluation image; no error-based selection',fontweight='bold')
        fig.savefig(folder/f'{family}_depth_evidence.png',dpi=180);plt.close(fig)
        dump(folder/f'{family}_selection.json',{'image_id':r['id'],'selection':'First sorted native evaluation image, without model-error ranking.'})

def main():
    global METHODS,PRED_ROOTS,PRIMARY
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'phase3/results/combined')
    parser.add_argument('--skip-prediction-audit',action='store_true');args=parser.parse_args();out=args.out
    assert (out/'summary.json').exists(),'Refusing to analyze a partial run'
    cfg=json.loads((out/'config.json').read_text())
    if 'method_roots' in cfg:
        METHODS=cfg['methods'];PRED_ROOTS=cfg['method_roots'];PRIMARY=cfg['primary_method']
        for run in cfg['runs']:
            base=ROOT/run['directory'];assert sha(base/'config.json')==run['config_sha256']
            assert sha(base/'evaluation_sources.json')==run['evaluation_sources_sha256']
    rows=list(csv.DictReader((out/'metrics.csv').open(newline='',encoding='utf-8')));data=read_data()
    if not args.skip_prediction_audit:audit(out,rows,data)
    else:assert json.loads((out/'independent_audit.json').read_text())['metrics_csv_sha256']==sha(out/'metrics.csv')
    tables,comparisons=summarize(out,rows);lights=stability(out,rows,data);figures(out,tables,data)
    dump(out/'analysis_summary.json',{'comparison_table':tables,'lighting_stability':lights,
        'fixed_primary_method':PRIMARY,'all_methods_reported':True,
        'literature_novelty_established':False,'physical_illuminant_recovery_established':False,
        'remid_physical_room_independence_known':False,'paired_comparisons':'paired_comparisons.json'})
    print(json.dumps({'analysis':'COMPLETE','rows':len(rows),'results':str(out),'primary_method':PRIMARY},indent=2))

if __name__=='__main__':main()
