"""Plot every declared primary/secondary result after a passing audit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HOME=Path(__file__).resolve().parent
OUT=HOME/'results/nyu_frame_v1'


def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='PASS'
    rows=json.loads((OUT/'summary.json').read_text())
    rows={r['condition']:r for r in rows if r['selection']=='fixed_final' and r['group']=='all'}
    order=['native','camera_real','camera_unitdepth','native_flat','camera_flat']
    labels=['Native XYZ','Canonical axes','Canonical axes\n+ median depth=1','Native flat','Canonical flat']
    plt.rcParams.update({'font.size':10,'font.family':'Arial'})
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    v=[rows[c]['mean'] for c in order]
    ax.bar(np.arange(5),v,color=['#64748b','#2563eb','#0f766e','#a8a29e','#d97706'])
    for i,x in enumerate(v):ax.text(i,x+.035,f'{x:.3f}',ha='center')
    ax.set_xticks(np.arange(5),labels)
    ax.set_ylabel('Mean illuminant recovery angular error (degrees)')
    ax.set_ylim(0,max(v)*1.22)
    ax.set_title('NYU + DIODE: fixed 10,000-epoch PCCC, official three-fold predictions\n324 captures; one seed/fold; old native baselines reused')
    ax.spines[['top','right']].set_visible(False)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'all_conditions.png',dpi=180);plt.close(fig)
    pairs=json.loads((OUT/'paired.json').read_text())
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for i,p in enumerate(pairs):
        a,b=p['conditional_capture_bootstrap95'];m=p['mean_deg']
        ax.errorbar(m,i,xerr=[[m-a],[b-m]],fmt='o',color='#0f766e' if p['role']=='primary' else '#64748b',capsize=4)
    ax.axvline(0,color='#a8a29e',lw=1)
    ax.set_yticks(range(len(pairs)),[p['contrast']+' ('+p['role']+')' for p in pairs])
    ax.invert_yaxis();ax.set_xlabel('Paired mean error difference (degrees); negative favors first condition')
    ax.set_title('All predeclared contrasts with conditional capture-bootstrap 95% intervals\nNot seed/room population intervals; no multiplicity adjustment')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'paired_contrasts.png',dpi=180);plt.close(fig)
    print('Saved all-condition and paired-contrast figures.')


if __name__=='__main__':main()

