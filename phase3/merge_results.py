"""Combine both complete, predeclared runs without selecting winners on test."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    runs=['phase3/results/study_v1','phase3/results/spatial_v1','phase3/results/spatial_controls']
    merged={};roots={};configs=[];sources={};manifests={};counts={};model_hashes={}
    for relative in runs:
        folder=ROOT/relative;cfg=json.loads((folder/'config.json').read_text())
        assert (folder/'summary.json').exists(),'Refusing partial runs'
        evaluation=json.loads((folder/'evaluation_sources.json').read_text())
        assert evaluation['configuration_sha256']==cfg['configuration_sha256']
        configs.append({'directory':relative,'config_sha256':sha(folder/'config.json'),
            'evaluation_sources_sha256':sha(folder/'evaluation_sources.json')})
        for field,destination in [('sources',sources),('manifests',manifests)]:
            for key,value in cfg[field].items():
                if key in destination:assert destination[key]==value
                destination[key]=value
        for method in cfg['variant_mode_equivariance_weight']:
            assert method not in roots
            roots[method]=relative;counts[method]=cfg.get('trainable_parameters',59699)
            model_hashes[method]=evaluation['checkpoint_sha256'][method]
        roots.setdefault('frozen',relative)
        rows=list(csv.DictReader((folder/'metrics.csv').open(newline='',encoding='utf-8')))
        for row in rows:
            key=tuple(row[k] for k in ['family','id','condition','method'])
            if key in merged:
                assert row['method']=='frozen'
                for metric in ['abs_rel','rmse_m','delta1']:assert abs(float(row[metric])-float(merged[key][metric]))<1e-10
            else:merged[key]=row
    methods=['frozen']+[m for m in roots if m!='frozen']
    expected=1639*len(methods);assert len(merged)==expected
    out=ROOT/'phase3/results/combined';out.mkdir(parents=True,exist_ok=True)
    config={'runs':configs,'sources':sources,'manifests':manifests,'method_roots':roots,
        'methods':methods,'method_parameter_counts':counts,'checkpoint_sha256':model_hashes,
        'primary_method':'spatial_geometry_eq','expected_rows':expected,
        'selection_note':'All variants retained. No ranking or model choice based on test scores.'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8',newline='\n')
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(next(iter(merged.values()))));writer.writeheader();writer.writerows(merged.values())
    (out/'summary.json').write_text(json.dumps({'complete':True,'rows':expected,'methods':methods},indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'combined_rows':expected,'methods':len(methods),'all_variants_retained':True},indent=2))

if __name__=='__main__':main()
