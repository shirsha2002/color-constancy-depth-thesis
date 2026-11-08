"""Freeze a small real DepthAWB experiment before training or test scoring."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from upstream import ROOT,identities,sha

OUT=ROOT/'phase4_pccc'
DATA=ROOT/'phase3/data/pccc'
SEED=20261109

def main():
    upstream=identities()
    path=OUT/'manifest.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        assert frozen['upstream']==upstream
        for record in frozen['samples']:
            for kind in ['cloud','label']:assert sha(ROOT/record[kind])==record[kind+'_sha256']
        assert sha(DATA/'archives/Depth_AWB_folds.csv')==frozen['fold_csv_sha256']
        print(json.dumps({'manifest_audit':'PASS','samples':len(frozen['samples']),'unchanged':True}));return
    rows=list(csv.DictReader((DATA/'archives/Depth_AWB_folds.csv').open(newline='')))
    assert len(rows)==185 and len({r['fn'] for r in rows})==185
    development=sorted([r['fn'] for r in rows if r['foldnum']!='0'])
    validation=set(np.random.default_rng(SEED).choice(development,size=25,replace=False))
    samples=[];seen={};zero_points=0
    for row in sorted(rows,key=lambda r:int(Path(r['fn']).stem)):
        cloud=DATA/'raw/RealAWB_pointcloud/new/pcd16'/row['fn']
        label=DATA/'raw/real_awb/label2'/row['fn']
        x=np.load(cloud,allow_pickle=False);y=np.load(label,allow_pickle=False)
        assert x.shape==(256,6) and y.shape==(3,) and np.isfinite(x).all() and np.isfinite(y).all()
        assert (x[:,2]<=0).all() and (x[:,3:]>=0).all() and (x[:,3:]<=1).all() and (y>0).all()
        invalid=np.abs(x[:,2])<1e-8
        assert (x[invalid,:3]==0).all()
        h=hashlib.sha256(x.tobytes()).hexdigest();assert h not in seen;seen[h]=row['fn']
        split='test' if row['foldnum']=='0' else ('validation' if row['fn'] in validation else 'train')
        zero_points+=int(invalid.sum())
        samples.append({'id':Path(row['fn']).stem,'official_fold':int(row['foldnum']),'split':split,
            'cloud':cloud.relative_to(ROOT).as_posix(),'cloud_sha256':sha(cloud),
            'label':label.relative_to(ROOT).as_posix(),'label_sha256':sha(label),
            'invalid_geometry_points':int(invalid.sum())})
    counts={s:sum(r['split']==s for r in samples) for s in ['train','validation','test']}
    assert counts=={'train':98,'validation':25,'test':62}
    manifest={'dataset':'DepthAWB original public PCCC release','client_date':'2026-11-09',
        'source':'https://github.com/xyxingx/Point-Cloud-Color-Constancy','upstream':upstream,
        'acquisition_manifest_sha256':sha(DATA/'acquisition_manifest.json'),
        'fold_csv_sha256':sha(DATA/'archives/Depth_AWB_folds.csv'),'split_seed':SEED,
        'selection':'All185officialsamples; official fold0 is final test; inner25validation from folds1+2 before training',
        'counts':counts,'points_per_sample':256,'zero_geometry_points_retained':zero_points,
        'geometry':'Native released normalized XYZ has negative Z; no metric-unit claim. Invalid zero XYZ retained.',
        'limitations':['Single official outer fold, not full3foldbenchmark.',
            'Unknown physical-scene identities; capture disjointness is not proven room independence.',
            'Original full-resolution RGB and sensor-depth registration not independently reconstructed.'],
        'samples':samples}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'frozen':str(path),'counts':counts,'zero_geometry_points':zero_points},indent=2))

if __name__=='__main__':main()
