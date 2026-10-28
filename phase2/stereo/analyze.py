"""Audit complete DTU results; create measured figures and unmasked 3D exports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

from geometry import camera,backproject,write_ply
from costs import METHODS

ROOT=Path(__file__).resolve().parents[2];PHASE=ROOT/'phase2'
DATA=PHASE/'data/dtu/source/SampleSet/MVS Data'
LABEL={'rgb_ssd':'RGB SSD','scalar_gain':'Scalar gain','diagonal_gain':'Diagonal gain',
       'gray_zncc':'Gray ZNCC','channel_zncc':'Channel ZNCC'}
COLORS=['#d87d35','#6491a2','#007c80','#9674a1','#2853a0']


def save(fig,path):
    fig.savefig(path,dpi=190,bbox_inches='tight',pad_inches=.15);plt.close(fig)


def audit(out,rows,protocol):
    checks=[];masks={}
    for f in sorted((out/'arrays').glob('*.npz')):
        rec=json.loads((out/'records'/f.with_suffix('.json').name).read_text())
        a=np.load(f);gt=a['gt'];mask=a['mask'];pred=a['depths']
        first=rec['metrics'][0];key=(first['scene'],first['reference_view'])
        if key in masks:
            if not np.array_equal(mask,masks[key]):raise ValueError('Scoring mask changes with lighting')
        else:masks[key]=mask
        finite=np.isfinite(pred)
        if not all(np.array_equal(finite[0],m) for m in finite):raise ValueError('Method support differs')
        for mi,method in enumerate(METHODS):
            error=pred[mi][mask].astype(np.float64)-gt[mask].astype(np.float64)
            metrics={'mae_mm':np.abs(error).mean(),'rmse_mm':np.sqrt((error**2).mean()),
                     'bad5_percent':100*(np.abs(error)>5).mean(),'bad10_percent':100*(np.abs(error)>10).mean()}
            original=rec['metrics'][mi]
            for k,v in metrics.items():
                if not np.isclose(v,original[k],rtol=3e-6,atol=3e-5):raise ValueError(f'Independent score mismatch {f.name} {method} {k}')
            checks.append({'case':f.stem,'method':method,'mae_mm_float64':float(metrics['mae_mm'])})
    expected=len(protocol['actual_scenes'])*len(protocol['actual_reference_views'])*len(protocol['actual_source_lights'])*len(METHODS)
    if len(checks)!=expected or len(rows)!=expected:raise ValueError(f'Incomplete complete-run evidence: {len(checks)}/{expected}')
    result={'audited_scores':len(checks),'method_support_identical':True,'lighting_masks_identical':True,
            'independent_float64_score_recalculation_passed':True,'expected_scores':expected,'checks':checks}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def group_plot(rows,out):
    rr=[r for r in rows if r['split']=='evaluation']
    refs=sorted({int(r['reference_view']) for r in rr})
    fig,axes=plt.subplots(1,len(refs),figsize=(13.4,4.3),sharey=True,squeeze=False)
    for ax,view in zip(axes[0],refs):
        for method,color in zip(METHODS,COLORS):
            selected=sorted([r for r in rr if int(r['reference_view'])==view and r['method']==method],key=lambda r:int(r['source_light']))
            ax.plot([int(r['source_light']) for r in selected],[float(r['mae_mm']) for r in selected],'-o',lw=2,color=color,label=LABEL[method])
        ax.axvspan(2.85,3.15,color='#dceee6',alpha=.7,zorder=0)
        ax.set_title(f'Reference view {view}',fontsize=12);ax.set_xticks(range(7));ax.set_xlabel('Source light ID (3 = matched diffuse)');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Depth MAE (mm; lower is better)')
    fig.suptitle('Real lighting changes and calibrated stereo: held-out DTU scene 6',fontweight='bold',fontsize=15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.09),frameon=False)
    fig.text(.5,-.14,'Same geometric support and fixed depth search for all costs. One held-out scene; curves are descriptive, not population estimates.',ha='center',fontsize=9,color='#657788')
    fig.tight_layout(rect=[0,.01,1,.93]);save(fig,out/'figures/heldout_lighting_mae.png')


def comparison(rows,out,ref=25):
    rr=[r for r in rows if r['split']=='evaluation' and int(r['reference_view'])==ref]
    arrays={}
    for light in [3,0]:
        f=next((out/'arrays').glob(f'scene006_ref{ref:03d}_*_light{light}.npz'))
        arrays[light]=np.load(f)
    fig,axes=plt.subplots(2,6,figsize=(18.5,7))
    for ri,light in enumerate([3,0]):
        a=arrays[light];mask=a['mask'];gt=a['gt']
        data=[gt]+[a['depths'][mi] for mi in range(len(METHODS))]
        titles=['STL reference']+[LABEL[m] for m in METHODS]
        for ci,(z,title) in enumerate(zip(data,titles)):
            ax=axes[ri,ci];im=ax.imshow(np.ma.masked_where(~mask,z),cmap='magma_r',vmin=250,vmax=1200)
            if ci:
                row=next(r for r in rr if r['method']==METHODS[ci-1] and int(r['source_light'])==light)
                title+=f'\nMAE {float(row["mae_mm"]):.2f} mm'
            else:title+='\nMatched 3/3' if light==3 else '\nMismatched 3/0'
            ax.set_title(title,fontsize=11);ax.axis('off')
    fig.subplots_adjust(left=.01,right=.965,top=.88,bottom=.07,wspace=.035,hspace=.24)
    cax=fig.add_axes([.975,.18,.008,.58]);cb=fig.colorbar(im,cax=cax);cb.set_label('Camera-Z depth (mm)')
    fig.suptitle(f'Held-out scene 6, reference {ref}: actual calibrated depth predictions',fontweight='bold',fontsize=16)
    fig.text(.015,.015,'Displayed on the shared evaluation domain. Identical depth color scale250-1200mm; structured-light reference holes are not interpolated.',fontsize=10,color='#657788')
    save(fig,out/'figures/heldout_depth_comparison.png')
    fig,axes=plt.subplots(2,5,figsize=(16.2,6.5))
    for ri,light in enumerate([3,0]):
        a=arrays[light]
        for mi,m in enumerate(METHODS):
            err=np.abs(a['depths'][mi]-a['gt']);ax=axes[ri,mi]
            im=ax.imshow(np.ma.masked_where(~a['mask'],err),cmap='inferno',vmin=0,vmax=100)
            ax.set_title(('Matched3/3 | ' if ri==0 else 'Mismatched3/0 | ')+LABEL[m],fontsize=11);ax.axis('off')
    fig.subplots_adjust(left=.01,right=.955,top=.88,bottom=.08,wspace=.035,hspace=.20)
    cax=fig.add_axes([.97,.18,.01,.58]);cb=fig.colorbar(im,cax=cax,ticks=[0,20,40,60,80,100]);cb.ax.set_yticklabels(['0','20','40','60','80','100+']);cb.set_label('Absolute depth error (mm)')
    fig.suptitle('Where real lighting changes damage geometry',fontweight='bold',fontsize=15)
    fig.text(.015,.015,'Same evaluation pixels for every panel; error colors share0-100mm(100+ saturated). No GT enters matching or gain estimation.',fontsize=10,color='#657788')
    save(fig,out/'figures/heldout_error_comparison.png')
    # The source photographs show a real light change on an unchanged view.
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.7))
    paths=[(25,3),(26,3),(26,0)]
    for ax,(v,l) in zip(axes,paths):
        im=cv2.imread(str(DATA/f'Rectified/scan6/rect_{v:03d}_{l}_r5000.png'))[:,:,::-1]
        ax.imshow(im);ax.set_title(f'View {v}, light {l}'+(' (reference)' if v==25 else ' (source)'),fontsize=11);ax.axis('off')
    fig.suptitle('Original photographs: geometry fixed, source lighting changed',fontweight='bold');fig.tight_layout();save(fig,out/'figures/heldout_real_inputs.png')
    # Fixed diagnostic points chosen from reference texture and valid geometry,
    # without selecting according to a method's outcome.
    a0=arrays[0];pts=a0['points_yx'][:3];fig,axes=plt.subplots(2,len(pts),figsize=(12,6),squeeze=False)
    for ri,light in enumerate([3,0]):
        a=arrays[light]
        for pi,(y,x) in enumerate(pts):
            ax=axes[ri,pi]
            for mi,color in enumerate(COLORS):
                c=a['curves'][mi,:,pi].copy();finite=np.isfinite(c)
                if not finite.any():continue
                lo,hi=np.min(c[finite]),np.max(c[finite]);c[finite]=(c[finite]-lo)/max(float(hi-lo),1e-9)
                c[~finite]=np.nan;ax.plot(a['planes_mm'],c,color=color,lw=1.5,label=LABEL[METHODS[mi]])
            ax.axvline(a['gt'][y,x],color='black',ls='--',lw=1.1,label='STL depth')
            ax.set_title(f'Pixel({x},{y}) | source light{light}',fontsize=10);ax.set_xlabel('Depth hypothesis(mm)');ax.set_ylabel('Cost, visual min-max normalization');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc='lower center',bbox_to_anchor=(.5,-.025),frameon=False,fontsize=9)
    fig.suptitle('Cost-volume diagnostics at preselected textured reference points',fontweight='bold');fig.tight_layout(rect=[0,.03,1,.95]);save(fig,out/'figures/heldout_cost_curves.png')


def inference_exports(out,protocol):
    target=out/'pointclouds_inference';target.mkdir(exist_ok=True)
    chosen=['rgb_ssd','scalar_gain','diagonal_gain','channel_zncc'];allpoints={};allcolors={}
    for f in sorted((out/'arrays').glob('scene006_*.npz')):
        if not (f.stem.endswith('_light0') or f.stem.endswith('_light3')):continue
        rec=json.loads((out/'records'/f.with_suffix('.json').name).read_text())
        ref=rec['metrics'][0]['reference_view'];light=rec['metrics'][0]['source_light'];a=np.load(f)
        cam=camera(DATA/f'Calibration/cal18/pos_{ref:03d}.txt',shape=(protocol['height'],protocol['width']),pixel_origin=protocol['pixel_origin'])
        for m in chosen:
            dep=a['depths'][METHODS.index(m)];valid=np.isfinite(dep)
            X=backproject(dep,cam)[valid];rgb=a['ref_rgb'][valid]
            write_ply(target/f'{f.stem}__{m}.ply',X,rgb)
            allpoints.setdefault((light,m),[]).append(X);allcolors.setdefault((light,m),[]).append(rgb)
    metadata=[]
    for key,pp in allpoints.items():
        points=np.concatenate(pp);color=np.concatenate(allcolors[key]);light,m=key
        # Duplicate removal uses only predicted coordinates, not GT.
        voxel=np.floor(points).astype(np.int32);_,keep=np.unique(voxel,axis=0,return_index=True)
        points=points[keep];color=color[keep]
        name=f'fused_scene006_light{light}__{m}.ply';write_ply(target/name,points,color)
        # Small deterministic preview suitable for sharing without all samples.
        rng=np.random.default_rng(20261028);idx=rng.choice(len(points),min(len(points),30000),replace=False)
        write_ply(target/name.replace('.ply','_preview30k.ply'),points[idx],color[idx])
        metadata.append({'file':name,'points':len(points),'voxel_mm':1,'gt_filter':False})
    (target/'README.json').write_text(json.dumps({'domain':'full finite inference support; no GT foreground, visibility or accuracy mask',
        'note':'Local matching may produce background/outlier points. These are algorithm outputs, not complete benchmark-quality meshes.',
        'exports':metadata},indent=2),encoding='utf-8')


def summary_note(rows,out,audit_info):
    rr=[r for r in rows if r['split']=='evaluation']
    def avg(m,mode):
        a=[float(r['mae_mm']) for r in rr if r['method']==m and r['lighting']==mode]
        return float(np.mean(a))
    lines=['# DTU real-light stereo: actual held-out results','',
        'One held-out scene (scene6), three reference views, six mismatched source-light conditions plus the matched diffuse control. All results below are descriptive; there are no population confidence intervals.','',
        '|Matching cost|Matched light MAE(mm)|Mismatched lights mean MAE(mm)|','|---|---:|---:|']
    for m in METHODS:lines.append(f'|{LABEL[m]}|{avg(m,"matched_diffuse"):.3f}|{avg(m,"mismatched"):.3f}|')
    rawm=avg('rgb_ssd','matched_diffuse');rawx=avg('rgb_ssd','mismatched')
    scal=avg('scalar_gain','mismatched');diag=avg('diagonal_gain','mismatched');ncc=avg('channel_zncc','mismatched')
    lines+=['',f'Raw RGB matching changes from {rawm:.3f}mm to {rawx:.3f}mm under mismatched lighting ({(rawx/rawm-1)*100:+.1f}%).',
        f'The diagonal candidate averages {diag:.3f}mm versus {scal:.3f}mm for the scalar ablation and {ncc:.3f}mm for established channelZNCC. '
        +('It does not outperform the scalar ablation.' if diag>=scal else 'It improves on the scalar ablation within this single-scene pilot.'),
        '',f'Independent float64 recomputation checked {audit_info["audited_scores"]} per-case/method score rows. Scoring masks are identical across lighting and output support is identical across methods.',
        '', 'Geometry was checked using supplied projection decomposition, numerical roundtrips and RGB/STL support overlays. References are z-buffered original STL point samples, not inferred depth or interpolated dense ground truth.',
        '', 'Important scope limits: one held-out object; fixed low resolution and finite depth planes; no SGM/global smoothness; directional light rather than measured illuminant-color ground truth. Known radiometric matching costs are baselines, not a new color-constancy contribution.',
        '', 'The pointclouds_inference directory contains full finite outputs without GT masking. The pointclouds directory and comparison panels use common evaluation support and are labeled accordingly.',
        '', 'See phase2/research/stereo_design_audit.md and the copied protocol.json for sources, equations and exact settings.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='main_v2');args=ap.parse_args()
    out=PHASE/'results/stereo'/args.run
    if not (out/'summary.json').exists():raise RuntimeError('Full run not complete; refusing to author results from partial data')
    protocol=json.loads((out/'protocol.json').read_text())
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'Arial','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    info=audit(out,rows,protocol);group_plot(rows,out);comparison(rows,out);inference_exports(out,protocol);summary_note(rows,out,info)
    print(json.dumps({'audit_rows':info['audited_scores'],'results':str(out/'RESULTS.md'),'figures':str(out/'figures')},indent=2))


if __name__=='__main__':main()
