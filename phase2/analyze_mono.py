"""Make scientific evidence figures from the completed frozen Vari experiment."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase2/results/mono';FIG=OUT/'figures'
NAMES={'frozen':'Frozen model','gray_world_preprocess':'GW preprocessing','task_adapter':'Task adapter',
       'achromatic_consistency':'Achromatic + consistency','cc_supervised':'Chromatic, depth only','cc_consistency':'Chromatic + consistency'}
COLORS=['#7B8792','#B88869','#1D4556','#7B79A0','#679A94','#007F7B']

def main():
    s=json.loads((OUT/'summary.json').read_text());FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    results=s['results'];methods=[r['method'] for r in results];labels=[NAMES[m] for m in methods]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,key,title in zip(axes,['abs_rel','lighting_instability'],['Depth accuracy: mean AbsRel','Lighting instability: normalized MAD']):
        vals=[r[key] for r in results];bars=ax.barh(range(6),vals,color=COLORS)
        ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.25);ax.set_title(title,fontweight='bold')
        for bar,value in zip(bars,vals):ax.text(value+max(vals)*.025,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center',fontsize=10)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_xlabel('Lower is better')
    fig.suptitle('Held-out room 485: 24 camera groups, 264 lighting images',fontweight='bold',fontsize=14)
    fig.tight_layout();fig.savefig(FIG/'mono_comparison.png',dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11.5,4.7))
    lights=sorted({r['lighting'] for r in s['by_lighting']})
    for i,method in enumerate(methods[2:],start=2):
        ys=[next(r['abs_rel'] for r in s['by_lighting'] if r['method']==method and r['lighting']==l) for l in lights]
        ax.plot(range(len(lights)),ys,'o-',color=COLORS[i],label=NAMES[method],lw=2)
    ax.set_xticks(range(len(lights)),[l.replace('_','\n') for l in lights],fontsize=8)
    ax.set_ylim(0,max(ax.get_ylim()[1],.5));ax.set_ylabel('Mean per-image AbsRel');ax.set_title('Learned variants under eleven rendered lighting conditions',fontweight='bold')
    ax.set_title('Learned variants under eleven rendered lighting conditions',fontweight='bold',pad=51)
    ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'lighting_accuracy.png',dpi=190);plt.close(fig)

    records=s['dataset']['images'];evaluation=[r for r in records if r['split']=='evaluation']
    group=sorted({r['view_group'] for r in evaluation})[0]
    chosen=sorted([r for r in evaluation if r['view_group']==group],key=lambda r:r['lighting'])
    r=chosen[0];gt=np.load(ROOT/r['depth']);valid=np.isfinite(gt)&(gt>.1)&(gt<9.99)
    fig=plt.figure(figsize=(11.8,7.0))
    grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.045],left=.02,right=.94,bottom=.04,top=.88,wspace=.12,hspace=.23)
    axes=np.array([[fig.add_subplot(grid[row,col]) for col in range(3)] for row in range(2)])
    axes[0,0].imshow(Image.open(ROOT/r['rgb']));axes[0,0].set_title('Input RGB / '+r['lighting'])
    panels=[(axes[0,1],gt,'Reference depth'),(axes[0,2],np.load(OUT/'predictions/frozen'/f'{r["id"]}.npy'),'Frozen model'),
            (axes[1,0],np.load(OUT/'predictions/task_adapter'/f'{r["id"]}.npy'),'Task adapter'),
            (axes[1,1],np.load(OUT/'predictions/cc_consistency'/f'{r["id"]}.npy'),'Chromatic + consistency')]
    for ax,array,title in panels:
        im=ax.imshow(np.clip(array,.1,10),vmin=.1,vmax=10,cmap='magma');ax.set_title(title)
    error=np.abs(np.clip(panels[-1][1],.1,10)-gt);error[~valid]=np.nan
    err=axes[1,2].imshow(error,vmin=0,vmax=2,cmap='inferno');axes[1,2].set_title('Chromatic absolute error (m)')
    for ax in axes.ravel():ax.axis('off')
    fig.colorbar(im,cax=fig.add_subplot(grid[0,3]),label='Depth (m), all depth maps')
    fig.colorbar(err,cax=fig.add_subplot(grid[1,3]),label='Error (m), 2+ saturated')
    fig.suptitle(f'Fixed first evaluation camera: {group}; no performance-based selection',fontweight='bold',fontsize=13)
    fig.savefig(FIG/'depth_evidence.png',dpi=180);plt.close(fig)

    indices=[0,len(chosen)//2,len(chosen)-1];fig,axes=plt.subplots(1,3,figsize=(11.5,3.2))
    for ax,i in zip(axes,indices):ax.imshow(Image.open(ROOT/chosen[i]['rgb']));ax.set_title(chosen[i]['lighting'].replace('_',' '),fontsize=10);ax.axis('off')
    fig.suptitle('Same camera and geometry, different rendered lighting',fontweight='bold',fontsize=13)
    fig.tight_layout();fig.savefig(FIG/'lighting_inputs.png',dpi=180);plt.close(fig)
    (FIG/'selection.json').write_text(json.dumps({'view_group':group,'image_id':r['id'],'lighting':r['lighting'],'selection':'First sorted evaluation camera and lighting, chosen without model-error ranking.'},indent=2))
    print(FIG)

if __name__=='__main__':main()
