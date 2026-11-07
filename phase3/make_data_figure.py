"""Illustrate first fixed ReMID capture without using prediction error selection."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    m=json.loads((ROOT/'phase3/data/remid/manifest.json').read_text())
    first=sorted({r['view_group'] for r in m['images']})[0]
    records=sorted([r for r in m['images'] if r['view_group']==first],key=lambda r:r['lighting'])
    chosen=[records[i] for i in [0,len(records)//2,len(records)-1]]
    plt.rcParams.update({'font.family':'Arial','font.size':11})
    fig=plt.figure(figsize=(13,3.6));grid=fig.add_gridspec(1,5,width_ratios=[1,1,1,1,.04],left=.01,right=.95,top=.81,bottom=.05,wspace=.08)
    for k,r in enumerate(chosen):
        ax=fig.add_subplot(grid[0,k]);ax.imshow(Image.open(ROOT/r['rgb']));ax.set_title('Captured light / '+r['lighting']);ax.axis('off')
    depth=np.load(ROOT/chosen[0]['depth']).astype(np.float32)*.001
    depth[~(np.isfinite(depth)&(depth>.1)&(depth<10))]=np.nan
    ax=fig.add_subplot(grid[0,3]);im=ax.imshow(depth,vmin=.1,vmax=10,cmap='magma');ax.set_title('Shared measured depth');ax.axis('off')
    fig.colorbar(im,cax=fig.add_subplot(grid[0,4]),label='Depth (m)')
    fig.suptitle('ReMID: captured real lighting, fixed camera and aligned reference',fontweight='bold',fontsize=14)
    out=ROOT/'phase3/results';out.mkdir(exist_ok=True)
    fig.savefig(out/'real_lighting_data.png',dpi=190);plt.close(fig)
    (out/'real_lighting_selection.json').write_text(json.dumps({'capture':first,'rgb_ids':[r['id'] for r in chosen],
        'selection':'First sorted capture; first,middle,last ordinal images; no prediction-error selection.',
        'source':m['source_paper']},indent=2),encoding='utf-8',newline='\n')

if __name__=='__main__':main()
