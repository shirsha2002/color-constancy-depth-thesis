"""Check actual quantized inputs, independently of any depth error."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase3/learning'))
import study

def main():
    cfg=json.loads((ROOT/'phase3/results/study_v1/config.json').read_text())
    folder=ROOT/'phase3/cache'/study.cache_spec(cfg)[:16]
    records=[r for r in study.inputs()['nyu'] if r['split']=='evaluation']
    errors=[];pre_errors=[];heads=[];exposure=[]
    for r in records:
        def item(c):return torch.load(folder/f'{r["key"]}__{c}.pt',map_location='cpu',weights_only=True)
        neutral=item('control');heads.append(neutral['intervention']['headroom'])
        def luminance(v):
            a=v['rgb_linear'][0].numpy().astype(np.float64)
            return float((a*np.array([.2126,.7152,.0722])[:,None,None]).sum(0).mean())
        base=luminance(neutral)
        for c in ['chroma_rb','chroma_green']:
            changed=item(c)
            assert changed['intervention']['headroom']==neutral['intervention']['headroom']
            assert changed['intervention']['clipped_fraction']==0
            pre_errors.append(abs(changed['intervention']['mean_linear_luminance']-neutral['intervention']['mean_linear_luminance']))
            errors.append(abs(luminance(changed)/base-1))
        changed=item('exposure');exposure.append(luminance(changed)/base)
    result={'intervention_audit':'PASS','scenes':len(records),'chromatic_pairs':len(errors),
        'pre_quantization_max_absolute_luminance_difference':max(pre_errors),
        'actual_cached_input_max_relative_luminance_drift':max(errors),
        'actual_cached_input_p95_relative_luminance_drift':float(np.quantile(errors,.95)),
        'exposure_ratio_min_max':[min(exposure),max(exposure)],
        'shared_headroom_min_max':[min(heads),max(heads)],'float_transform_clipping_fraction':0.,
        'note':'Actual-input estimates use area-pooled linearized8bitRGB. Prequantized chromatic arms match luminance; residual quantization drift is measured rather than claimed absent.'}
    assert result['pre_quantization_max_absolute_luminance_difference']<1e-10
    study.dump(ROOT/'phase3/results/intervention_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
