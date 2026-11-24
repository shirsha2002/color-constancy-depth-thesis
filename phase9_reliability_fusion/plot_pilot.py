"""Plot all fixed-final pilot outcomes after the independent audit passes."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HOME=Path(__file__).resolve().parent
OUT=HOME/'pilot_v1'


def main():
    audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='PASS'
    rows=json.loads((OUT/'summary.json').read_text())
    pairs=json.loads((OUT/'paired.json').read_text())
    methods=['rgb','ordinary','mask','learned','learned_clean','learned_no_consistency','pccc']
    conditions=['clean','noise10','noise30','drop50','shuffle']
    labels=['RGB only','Ordinary fusion','Mask fusion','Learned gate','Gate without damage training','Gate without consistency','Original PCCC']
    matrix=np.array([[next(r['mean'] for r in rows if r['method']==m and r['selection']=='fixed_final' and r['condition']==c) for c in conditions] for m in methods])
    plt.rcParams.update({'font.size':10,'font.family':'Arial'})
    fig,ax=plt.subplots(figsize=(9.8,5),layout='constrained')
    im=ax.imshow(matrix,cmap='YlOrRd',aspect='auto',vmin=0)
    for i in range(len(methods)):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='black' if matrix[i,j]<matrix.max()*.65 else 'white')
    ax.set_yticks(range(7),labels);ax.set_xticks(range(5),['Clean','10% noise','30% noise','50% depth missing','XYZ shuffled'])
    ax.set_title('ETH3D reliability fusion pilot - all methods and test conditions\n100 epochs; three official folds; one seed/fold; fixed-final checkpoints')
    fig.colorbar(im,ax=ax,label='Mean recovery angular error (degrees; lower is better)')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'all_methods.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    for i,p in enumerate(pairs):
        a,b=p['conditional_capture_bootstrap95'];mean=p['mean_deg']
        ax.errorbar(mean,i,xerr=[[mean-a],[b-mean]],fmt='o',capsize=5,color='#0f766e')
    ax.axvline(0,color='#64748b',lw=1)
    ax.set_yticks([0,1],['Clean inputs','Per-capture mean of four damage conditions'])
    ax.invert_yaxis();ax.set_xlabel('Learned gate minus mask fusion error (degrees); negative favors learned gate')
    ax.set_title('Both predeclared primary endpoints\nConditional capture-bootstrap 95% intervals; not seed/room intervals; no multiplicity adjustment')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'primary_contrasts.png',dpi=180);plt.close(fig)
    print('Saved all-method and both-primary-endpoint figures.')


if __name__=='__main__':main()
