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
    action='reduced' if gain>=0 else 'increased'
    change=100*(best['clean_abs_rel']/baseline['clean_abs_rel']-1)
    clean_action='increased' if change>=0 else 'reduced'
    msg=f'''Dear Professor, I would like to share my first experimental update on my thesis, "{s['thesis_title']}."

1. I set up and ran the official Depth Anything V2 Small indoor metric-depth model on my laptop. I evaluated {s['runtime']['evaluation_images']} real NYU Depth V2 RGB-depth pairs, one image per held-out scene, with {s['runtime']['development_images']} separate development scenes. The clean-image baseline has mean AbsRel {baseline['clean_abs_rel']:.4f} and RMSE {baseline['clean_rmse_m']:.3f} m.

2. I tested warm, cool, and green color casts at two strengths while keeping each scene and its reference depth fixed. Across these six synthetic conditions, the model without correction has mean AbsRel {baseline['corrupted_abs_rel']:.4f}.

3. I compared Gray-World and White-Patch correction and a fixed half-strength Gray-World variant. The lowest mean color-cast error among these tested methods came from {NAMES[best['method']]}, with AbsRel {best['corrupted_abs_rel']:.4f} and RMSE {best['corrupted_rmse_m']:.3f} m. This {action} mean AbsRel by {abs(gain):.1f}% relative to the same color-cast images without correction.

4. Correction also has a trade-off: this method {clean_action} error on clean images by {abs(change):.1f}%, and its effect varies across scenes. I included the accuracy table and selected helpful/failure examples in the slides and report. My next step is to study when correction should be applied, then test a second model and more realistic lighting changes.

These are results from a small controlled pilot with a pretrained model, using the same depth mask and no fitted depth scale. The code, sample IDs, raw metrics, and run configuration are saved for reproduction.

Shirsha Mitra
Student ID:2023521460130
'''
    out=ROOT/'output';out.mkdir(exist_ok=True)
    (out/'QQ_Update_English.txt').write_text(msg,encoding='utf-8')
    print(msg)

if __name__=='__main__':main()
