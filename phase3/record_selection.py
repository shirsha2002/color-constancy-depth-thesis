"""Record validation-only selection; never open a final-test score table here."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    rows=[]
    for run in ['study_v1','spatial_v1','spatial_controls']:
        folder=ROOT/'phase3/results'/run;cfg=json.loads((folder/'config.json').read_text())
        for method in cfg['variant_mode_equivariance_weight']:
            history=json.loads((folder/f'{method}_training.json').read_text())
            rows.append({'method':method,'run':run,'validation_absrel':history['validation_abs_rel'],
                'selected_step':history['selected_step'],'checkpoint_sha256':sha(folder/'models'/f'{method}.pt')})
    assert len(rows)==15
    best=min(rows,key=lambda r:r['validation_absrel'])
    result={'recorded_utc':datetime.now(timezone.utc).isoformat(),'selection_inputs':'Training histories and NYU validation scores only; no final test CSV or summary opened',
        'fixed_hypothesis_primary':'spatial_geometry_eq','validation_selected_recipe':best,
        'all_validation_recipes':rows,'note':'A validation-selected control is not automatically a successful CC method. Both the fixed hypothesis and selection are reported.'}
    path=ROOT/'phase3/results/development_selection.json'
    if path.exists():
        original=json.loads(path.read_text());assert original['all_validation_recipes']==rows
    else:path.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'validation_selection_recorded':True,'recipes':len(rows),'selected':best['method']},indent=2))

if __name__=='__main__':main()
