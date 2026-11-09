"""Freeze complete DepthAWB outer folds and paper-budget interpretation."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'phase5_pccc'
sys.path.insert(0,str(ROOT/'phase4_pccc'))
import upstream

def main():
    old=json.loads((ROOT/'phase4_pccc/manifest.json').read_text())
    samples=[]
    for r in old['samples']:
        for kind in ['cloud','label']:assert upstream.sha(ROOT/r[kind])==r[kind+'_sha256']
        samples.append({k:v for k,v in r.items() if k!='split'})
    source={'dataset':'Original DepthAWB PCCC release','client_date':'2026-11-09',
        'upstream':upstream.identities(),'samples':samples,'points_per_cloud':256,
        'folds':{str(f):{'train_ids':[r['id'] for r in samples if r['official_fold']!=f],
                         'test_ids':[r['id'] for r in samples if r['official_fold']==f]} for f in [0,1,2]},
        'scope':'All185samples and all3officialfolds; one benchmark, not all paper datasets/resolutions',
        'known_prior_access':'Allfolds participated in earlier pilot data roles. This is benchmark reproduction, not fresh independent confirmation.'}
    path=HOME/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==source
    else:path.write_text(json.dumps(source,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'manifest':'PASS','folds':{f:{k:len(v) for k,v in item.items()} for f,item in source['folds'].items()},'sha256':upstream.sha(path)},indent=2))

if __name__=='__main__':main()
