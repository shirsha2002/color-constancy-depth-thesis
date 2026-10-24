"""Write a concise professor update using verified completed run values only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAMES={'none':'no correction','gray_world':'Gray-World','white_patch':'White-Patch','gray_world_half':'half-strength Gray-World'}

def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    r={a['method']:a for a in s['results']};baseline=r['none']
    best=min((a for a in s['results'] if a['method']!='none'),key=lambda a:a['corrupted_abs_rel'])
    gain=100*(1-best['corrupted_abs_rel']/baseline['corrupted_abs_rel'])
    change=100*(best['clean_abs_rel']/baseline['clean_abs_rel']-1)
    clean_action='increased' if change>=0 else 'reduced'
    boot=next(b for b in s['bootstrap'] if b['method']==best['method'])
    uncertainty='The paired 95% confidence interval includes zero, so this pilot does not establish an average improvement.' if boot['ci95_low']<=0<=boot['ci95_high'] else 'The report includes the paired 95% confidence interval and per-scene effects.'
    msg=f'''Dear Professor, here is my first progress update on my color-constancy-assisted monocular depth estimation thesis.

1. I set up and ran the official Depth Anything V2 Small indoor metric-depth model on my laptop. I evaluated {s['runtime']['evaluation_images']} real NYU Depth V2 RGB-depth pairs, one image per held-out scene, with {s['runtime']['development_images']} separate development scenes. The clean-image baseline has mean AbsRel {baseline['clean_abs_rel']:.4f} and mean per-image RMSE {baseline['clean_rmse_m']:.3f} m.

2. I tested warm, cool, and green color casts at two strengths, keeping the scene and reference depth fixed. I compared no correction, Gray-World, White-Patch, and fixed half-strength Gray-World. The complete run saved {s['runtime']['unique_predictions']:,} numerical depth predictions.

3. The main finding is that correction helps some scenes and harms others. {NAMES[best['method']]} improved {boot['improved_images']} of {boot['n_images']} scene averages, but its mean AbsRel across the six casts was {best['corrupted_abs_rel']:.4f}, compared with {baseline['corrupted_abs_rel']:.4f} without correction ({abs(gain):.1f}% {'lower' if gain>=0 else 'higher'}). {uncertainty}

4. I also tested correction on clean images. This method {clean_action} their mean error by {abs(change):.1f}%, and its effect varies across scenes. The slides and report include the comparison table and selected helpful/failure cases. Next, I plan to study when correction should be applied, then test a second model and more realistic lighting changes.

These are results from a small controlled pilot with a pretrained model, using the same depth mask and no fitted depth scale. The code, sample IDs, raw metrics, and run configuration are saved for reproduction.

Shirsha Mitra
Student ID: 2023521460130
'''
    out=ROOT/'output';out.mkdir(exist_ok=True)
    (out/'QQ_Update_English.txt').write_text(msg,encoding='utf-8')
    print(msg)

if __name__=='__main__':main()
