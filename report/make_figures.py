"""Scientific figures drawn exclusively from the completed real pilot outputs.

Uses no model inference. Per-image comparisons preserve pairing across the six
casts. Qualitative examples are explicitly selected as extremes, not typical
or randomly chosen cases. Every depth panel shares the same 0.1-10 m scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from science import CASTS, METHODS, transform

NAVY='#13263B'; TEAL='#007C80'; ORANGE='#D87D35'; GREY='#657788'
COLORS={'none':ORANGE,'gray_world':TEAL,'white_patch':'#8565A0','gray_world_half':'#3676A5'}
LABELS={'none':'No correction','gray_world':'Gray-World','white_patch':'White-Patch',
        'gray_world_half':'Gray-World (0.5)'}
SHORT={'none':'None','gray_world':'Gray-World','white_patch':'White-Patch',
       'gray_world_half':'GW (0.5)'}
EVAL={'evaluation','test','val','validation'}


def style():
    plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.titlesize':14,
        'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#B8C6CF','axes.labelcolor':NAVY,'text.color':NAVY,
        'xtick.color':GREY,'ytick.color':GREY,'figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','axes.titleweight':'bold'})


def save(fig,path):
    fig.savefig(path,dpi=210,bbox_inches='tight',pad_inches=.17)
    plt.close(fig)


def load_results():
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    with (ROOT/'results/metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['abs_rel','rmse_m','mae_m','delta1','inference_seconds']:
            r[k]=float(r[k])
    rows=[r for r in rows if r['split'] in EVAL]
    if not rows:raise ValueError('No evaluation rows; refusing to draw development results as evaluation.')
    expected=len({r['image_id'] for r in rows})*len(CASTS)*len(METHODS)
    if len(rows)!=expected:
        raise ValueError(f'Incomplete rows: {len(rows)}, expected {expected}')
    return summary,rows


def mean_metric(rows,condition,method,metric='abs_rel'):
    rr=[r[metric] for r in rows if r['condition']==condition and r['method']==method]
    if not rr:raise ValueError((condition,method))
    return float(np.mean(rr))


def condition_chart(rows,out):
    fig,axes=plt.subplots(1,3,figsize=(11.3,5),sharey=True)
    for ax,cast in zip(axes,['warm','cool','green']):
        for method in METHODS:
            vals=[mean_metric(rows,c,method) for c in ['clean',cast+'_mild',cast+'_strong']]
            ax.plot([0,1,2],vals,'o-',lw=2.3,ms=6,color=COLORS[method],label=LABELS[method])
        ax.set_xticks([0,1,2],['Clean','Mild','Strong'])
        ax.set_title(cast.capitalize()+' cast',pad=11)
        ax.grid(axis='y',alpha=.22)
        ax.set_xlabel('Synthetic cast strength',labelpad=8)
    axes[0].set_ylabel('Mean AbsRel (lower is better)')
    fig.suptitle('Depth error changes with color-cast type and strength',fontsize=16,fontweight='bold',y=1.015)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.055),frameon=False,fontsize=11)
    fig.text(.5,-.095,'Same clean baseline repeated for context; averages use evaluation images only.',ha='center',fontsize=10,color=GREY)
    fig.tight_layout(rect=[0,.04,1,.95])
    save(fig,out/'condition_absrel.png')


def method_chart(rows,out):
    fig,axes=plt.subplots(1,2,figsize=(11.3,5))
    rng=np.random.default_rng(20261023)
    ids=sorted({r['image_id'] for r in rows})
    for ax,corrupt in zip(axes,[True,False]):
        vals=[]; lows=[]; highs=[]
        for method in METHODS:
            per_image=[]
            for iid in ids:
                v=[r['abs_rel'] for r in rows if r['image_id']==iid and r['method']==method
                   and ((r['condition']!='clean') if corrupt else (r['condition']=='clean'))]
                per_image.append(float(np.mean(v)))
            v=np.asarray(per_image)
            b=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)
            avg=float(v.mean());lo,hi=np.percentile(b,[2.5,97.5])
            vals.append(avg);lows.append(avg-float(lo));highs.append(float(hi)-avg)
        bars=ax.bar(np.arange(4),vals,color=[COLORS[m] for m in METHODS],width=.63,
                    yerr=[lows,highs],capsize=4,error_kw={'elinewidth':1.3,'ecolor':NAVY})
        for b,v,h in zip(bars,vals,highs):
            ax.text(b.get_x()+b.get_width()/2,v+h+.007,f'{v:.3f}',ha='center',fontsize=12,fontweight='bold')
        ax.set_xticks(range(4),['None','Gray-\nWorld','White-\nPatch','GW\n(0.5)'])
        ax.set_ylim(0,max(np.array(vals)+np.array(highs))*1.22)
        ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
        ax.set_title('Six synthetic casts, equally weighted' if corrupt else 'Clean inputs, same evaluation scenes',fontsize=13,pad=12)
    axes[0].set_ylabel('Mean AbsRel (lower is better)')
    fig.suptitle('Correction effects on cast and clean inputs',fontsize=16,fontweight='bold',y=1.01)
    fig.text(.5,-.035,'Bars: mean per-image error. Whiskers: 95% image-bootstrap intervals (10,000 resamples).',
             ha='center',fontsize=10,color=GREY)
    fig.tight_layout(rect=[0,.015,1,.95])
    save(fig,out/'method_comparison.png')


def load_manifest(path):
    p=json.loads(path.read_text(encoding='utf-8'))
    records=p if isinstance(p,list) else p.get('images',p.get('records',p.get('samples',[])))
    return {str(r.get('id',r.get('image_id'))):r for r in records}


def read_record(rec):
    rgb=np.asarray(Image.open(ROOT/rec.get('rgb',rec.get('rgb_path'))).convert('RGB'),dtype=np.float32)/255
    depth=np.load(ROOT/rec.get('depth',rec.get('depth_path'))).astype(np.float32)
    if rgb.shape[:2]!=depth.shape:raise ValueError('Unaligned RGB and reference depth')
    return rgb,depth


def pred(iid,cond,method):
    path=ROOT/'results/predictions'/f'{iid}__{cond}__{method}.npy'
    p=np.load(path).astype(np.float32)
    return np.clip(p,.1,10)


def mask_gt(gt):
    h,w=gt.shape
    m=np.zeros_like(gt,dtype=bool)
    m[round(45*h/480):round(471*h/480),round(41*w/640):round(601*w/640)]=True
    return m&np.isfinite(gt)&(gt>.1)&(gt<10)


def plot_image(ax,array,title,depth=False,error=False,mask=None):
    if depth or error:
        data=np.ma.masked_invalid(array)
        if mask is not None:data=np.ma.masked_where(~mask,data)
        cmap=plt.get_cmap('inferno' if error else 'magma_r').copy()
        cmap.set_bad('#E5E9EC')
        im=ax.imshow(data,cmap=cmap,vmin=0 if error else .1,vmax=3 if error else 10)
    else:im=ax.imshow(np.clip(array,0,1))
    ax.set_title(title,fontsize=12,pad=6)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    return im


def select_cases(rows):
    ids=sorted({r['image_id'] for r in rows})
    lookup={(r['image_id'],r['condition'],r['method']):r for r in rows}
    diffs={}
    for iid in ids:
        diffs[iid]=float(np.mean([lookup[(iid,c,'none')]['abs_rel']-lookup[(iid,c,'gray_world')]['abs_rel'] for c in CASTS if c!='clean']))
    choices={}
    for key,iid in [('best',max(ids,key=lambda i:diffs[i])),('failure',min(ids,key=lambda i:diffs[i]))]:
        choose=max if key=='best' else min
        c=choose((c for c in CASTS if c!='clean'),key=lambda c:lookup[(iid,c,'none')]['abs_rel']-lookup[(iid,c,'gray_world')]['abs_rel'])
        # Image selection uses all six casts. The best example then displays
        # its strongest benefit; the failure example displays its weakest.
        a=lookup[(iid,c,'none')];b=lookup[(iid,c,'gray_world')]
        choices[key]={'image_id':iid,'condition':c,'mean_six_cast_absrel_reduction':diffs[iid],
            'abs_rel_none':a['abs_rel'],'abs_rel_gray_world':b['abs_rel'],
            'rmse_none_m':a['rmse_m'],'rmse_gray_world_m':b['rmse_m'],
            'displayed_absrel_reduction':a['abs_rel']-b['abs_rel'],
            'clean_absrel':lookup[(iid,'clean','none')]['abs_rel'],
            'selection':('Extreme image by mean six-cast Gray-World AbsRel reduction; displayed cast is '
                         +('most favorable' if key=='best' else 'least favorable')+' within that image.')}
    return choices


def qualitative(case,rec,out,name):
    iid,cond=case['image_id'],case['condition']
    rgb,gt=read_record(rec)
    cast,_=transform(rgb,cond,'none');corrected,_=transform(rgb,cond,'gray_world')
    p0=pred(iid,cond,'none');p1=pred(iid,cond,'gray_world')
    valid=mask_gt(gt)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    plot_image(axes[0,0],cast,'Synthetic '+cond.replace('_',' '))
    gtplot=plot_image(axes[0,1],gt,'Reference depth (m)',depth=True,mask=np.isfinite(gt)&(gt>.1)&(gt<10))
    plot_image(axes[0,2],p0,f'No correction | AbsRel {case["abs_rel_none"]:.3f}',depth=True)
    plot_image(axes[0,3],p1,f'Gray-World | AbsRel {case["abs_rel_gray_world"]:.3f}',depth=True)
    plot_image(axes[1,0],rgb,'Original RGB')
    plot_image(axes[1,1],corrected,'Corrected RGB')
    errplot=plot_image(axes[1,2],np.abs(p0-gt),'No correction | absolute error (m)',error=True,mask=valid)
    plot_image(axes[1,3],np.abs(p1-gt),'Gray-World | absolute error (m)',error=True,mask=valid)
    scene=str(rec.get('scene',rec.get('scene_name','unknown')))
    case['scene']=scene
    label='Highest mean six-cast benefit' if name=='best' else ('Lowest mean six-cast benefit' if case['mean_six_cast_absrel_reduction']>=0 else 'Mean six-cast failure')
    fig.suptitle(f'{label} | {iid} | {scene}',fontsize=16,fontweight='bold',y=.99)
    fig.subplots_adjust(left=.015,right=.958,top=.91,bottom=.13,wspace=.06,hspace=.27)
    cax=fig.add_axes([.972,.55,.009,.31]);cbar=fig.colorbar(gtplot,cax=cax,ticks=[.1,2,4,6,8,10]);cbar.ax.tick_params(labelsize=9);cbar.set_label('Depth (m)',fontsize=10)
    eax=fig.add_axes([.972,.145,.009,.31]);ebar=fig.colorbar(errplot,cax=eax,ticks=[0,1,2,3]);ebar.ax.set_yticklabels(['0','1','2','3+']);ebar.ax.tick_params(labelsize=9);ebar.set_label('Abs. error (m)',fontsize=10)
    cast_label='most favorable' if name=='best' else 'least favorable'
    fig.text(.015,.065,f'Selected extreme, not a typical example. Displayed cast is the {cast_label} for Gray-World within this image.',fontsize=10,color=GREY)
    fig.text(.015,.025,'Depth scale: 0.1-10 m for every reference/prediction. Error: 0-3 m (3+ saturated); gray areas excluded from scoring.',fontsize=10,color=GREY)
    save(fig,out/f'qualitative_{name}.png')


def extras(case,rec,out):
    iid=case['image_id'];rgb,gt=read_record(rec);p=pred(iid,'clean','none')
    depthcmap=plt.get_cmap('magma_r')
    color=(depthcmap(Normalize(.1,10)(p))[...,:3]*255).astype(np.uint8)
    Image.fromarray(color).save(out/'cover_depth.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    plot_image(axes[0],rgb,'Input: original RGB')
    im=plot_image(axes[1],gt,'Reference depth',depth=True,mask=np.isfinite(gt)&(gt>.1)&(gt<10))
    plot_image(axes[2],p,f'Baseline depth | AbsRel {case["clean_absrel"]:.3f}',depth=True)
    fig.subplots_adjust(wspace=.04,left=.01,right=.94,top=.86,bottom=.03)
    ca=fig.add_axes([.955,.13,.012,.65]);cb=fig.colorbar(im,cax=ca,ticks=[.1,2,4,6,8,10]);cb.set_label('Depth (m)',fontsize=10);cb.ax.tick_params(labelsize=9)
    save(fig,out/'baseline_triptych.png')
    fig,axes=plt.subplots(1,4,figsize=(15,3.5))
    for ax,c,label in zip(axes,['clean','warm_strong','cool_strong','green_strong'],['Original','Warm cast','Cool cast','Green cast']):
        a,_=transform(rgb,c,'none');plot_image(ax,a,label)
    fig.subplots_adjust(wspace=.025,left=.01,right=.99,top=.86,bottom=.03)
    save(fig,out/'illumination_examples.png')


def report_variants(cases,records,out):
    """Larger labels and fewer panels for the narrower printed-page format."""
    case=cases['failure'];iid=case['image_id'];cond=case['condition']
    rgb,gt=read_record(records[iid]);cast,_=transform(rgb,cond,'none')
    p0=pred(iid,cond,'none');p1=pred(iid,cond,'gray_world');valid=mask_gt(gt)
    fig,axes=plt.subplots(2,3,figsize=(10.8,5.7))
    plot_image(axes[0,0],cast,'Input: synthetic '+cond.replace('_',' '))
    dm=plot_image(axes[0,1],p0,'No correction | AbsRel '+f'{case["abs_rel_none"]:.3f}',depth=True)
    plot_image(axes[0,2],p1,'Gray-World | AbsRel '+f'{case["abs_rel_gray_world"]:.3f}',depth=True)
    plot_image(axes[1,0],gt,'Reference depth',depth=True,mask=np.isfinite(gt)&(gt>.1)&(gt<10))
    em=plot_image(axes[1,1],np.abs(p0-gt),'No correction | absolute error',error=True,mask=valid)
    plot_image(axes[1,2],np.abs(p1-gt),'Gray-World | absolute error',error=True,mask=valid)
    fig.subplots_adjust(left=.01,right=.94,top=.93,bottom=.02,wspace=.05,hspace=.20)
    dc=fig.add_axes([.96,.55,.012,.32]);cb=fig.colorbar(dm,cax=dc,ticks=[.1,2,4,6,8,10]);cb.ax.tick_params(labelsize=9);cb.set_label('Depth (m)',fontsize=10)
    ec=fig.add_axes([.96,.08,.012,.32]);cb=fig.colorbar(em,cax=ec,ticks=[0,1,2,3]);cb.ax.set_yticklabels(['0','1','2','3+']);cb.ax.tick_params(labelsize=9);cb.set_label('Error (m)',fontsize=10)
    save(fig,out/'qualitative_failure_report.png')
    case=cases['best'];iid=case['image_id'];rgb,gt=read_record(records[iid]);p=pred(iid,'clean','none')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0))
    plot_image(axes[0],rgb,'Actual RGB input')
    dm=plot_image(axes[1],p,'Saved pretrained depth prediction',depth=True)
    fig.subplots_adjust(left=.01,right=.94,top=.89,bottom=.02,wspace=.04)
    dc=fig.add_axes([.96,.12,.012,.66]);cb=fig.colorbar(dm,cax=dc,ticks=[.1,2,4,6,8,10]);cb.ax.tick_params(labelsize=9);cb.set_label('Depth (m)',fontsize=10)
    save(fig,out/'cover_pair.png')
    return {'failure_report':'results/figures/qualitative_failure_report.png','cover_pair':'results/figures/cover_pair.png'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/nyu_pilot/manifest.json')
    ap.add_argument('--report-only',action='store_true',help='Only add print-friendly variants; preserve existing slide figures.')
    args=ap.parse_args()
    style();out=ROOT/'results/figures';out.mkdir(parents=True,exist_ok=True)
    summary,rows=load_results();records=load_manifest(args.manifest)
    if args.report_only:
        path=out/'figure_manifest.json'
        mapping=json.loads(path.read_text(encoding='utf-8'))
        mapping.update(report_variants(mapping['selection'],records,out))
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(mapping,indent=2),encoding='utf-8');temp.replace(path)
        print(json.dumps(mapping,indent=2));return
    condition_chart(rows,out);method_chart(rows,out)
    cases=select_cases(rows)
    for key,case in cases.items():
        if case['image_id'] not in records:raise KeyError(case['image_id'])
        qualitative(case,records[case['image_id']],out,key)
    extras(cases['best'],records[cases['best']['image_id']],out)
    shutil.copyfile(out/'qualitative_failure.png',out/'failure.png')
    mapping={k:'results/figures/'+v for k,v in {
        'cover':'cover_depth.png','illumination':'illumination_examples.png',
        'best':'qualitative_best.png','failure':'qualitative_failure.png',
        'baseline':'baseline_triptych.png','condition':'condition_absrel.png',
        'method':'method_comparison.png'}.items()}
    mapping['selection']=cases
    mapping.update(report_variants(cases,records,out))
    (out/'figure_manifest.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print(json.dumps(mapping,indent=2))


if __name__=='__main__':main()
