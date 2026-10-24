"""Independently audit source hashes, paired coverage and saved numeric scores."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'data/nyu_pilot/manifest.json').read_text(encoding='utf-8'))
    with (ROOT/'results/metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    images=manifest['images'];by_id={r['id']:r for r in images}
    dev={r['scene'] for r in images if r['split']=='development'}
    test={r['scene'] for r in images if r['split']=='evaluation'}
    assert len(dev)==8 and len(test)==32 and not dev&test,'Scene split differs from declared protocol'
    assert len(by_id)==len(images)==40
    split_file=ROOT/'data/nyu_pilot/splits.mat'
    splits=loadmat(split_file)
    official_ids={key:set(np.ravel(splits[key]).astype(int).tolist()) for key in ['trainNdxs','testNdxs']}
    assert len(official_ids['trainNdxs'])==795 and len(official_ids['testNdxs'])==654
    assert not official_ids['trainNdxs']&official_ids['testNdxs']
    assert hashlib.sha256(split_file.read_bytes()).hexdigest()==manifest['splits_mat_sha256']
    scene_labels=json.loads((ROOT/'data/nyu_pilot/official_scenes.json').read_text(encoding='utf-8'))
    for record in images:
        index=record['original_matlab_id']
        split_key='trainNdxs' if record['split']=='development' else 'testNdxs'
        assert index in official_ids[split_key]
        assert scene_labels[index-1]==record['scene']
    conditions=list(summary['protocol']['casts']);methods=summary['protocol']['methods']
    expected=set(itertools.product(by_id,conditions,methods))
    observed=[(r['image_id'],r['condition'],r['method']) for r in rows]
    assert len(observed)==len(set(observed)) and set(observed)==expected,'Incomplete/duplicated comparisons'
    depths={};hashes=0
    for record in images:
        for key in ['rgb','depth']:
            p=ROOT/record[key]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record[key+'_sha256'],f'Source hash mismatch:{p}'
            hashes+=1
        depths[record['id']]=np.load(ROOT/record['depth']).astype(np.float64)
    errors={k:0. for k in ['abs_rel','rmse_m','mae_m','delta1']}
    lower=summary['protocol']['evaluation_min_depth_m'];upper=summary['protocol']['evaluation_max_depth_m']
    top,bottom,left,right=summary['protocol']['crop']['at_480x640']
    fingerprints=set()
    for row in rows:
        image_id,condition,method=row['image_id'],row['condition'],row['method']
        stem=f'{image_id}__{condition}__{method}'
        pred=np.load(ROOT/'results/predictions'/f'{stem}.npy').astype(np.float64)
        gt=depths[image_id]
        assert pred.shape==gt.shape==(480,640) and np.isfinite(pred).all()
        crop=np.zeros(gt.shape,dtype=bool);crop[top:bottom,left:right]=True
        valid=crop&np.isfinite(gt)&(gt>lower)&(gt<upper)
        assert int(valid.sum())==int(row['valid_pixels'])
        g=gt[valid];p=np.clip(pred[valid],lower,upper);diff=p-g
        independent=dict(abs_rel=np.mean(np.abs(diff)/g),rmse_m=np.sqrt(np.mean(diff*diff)),
                         mae_m=np.mean(np.abs(diff)),delta1=np.mean(np.maximum(p/g,g/p)<1.25))
        for key,value in independent.items():
            errors[key]=max(errors[key],abs(float(value)-float(row[key])))
            assert errors[key]<1e-10,f'Saved score mismatch:{stem},{key}'
        meta=json.loads((ROOT/'results/predictions'/f'{stem}.json').read_text())
        fingerprints.add(meta['configuration_sha256'])
    config=json.loads((ROOT/'results/run_config.json').read_text())
    assert fingerprints=={config['configuration_sha256']}
    for result in summary['results']:
        selected=[r for r in rows if r['split']=='evaluation' and r['condition']!='clean' and r['method']==result['method']]
        for raw,aggregate in [('abs_rel','corrupted_abs_rel'),('rmse_m','corrupted_rmse_m'),('delta1','corrupted_delta1')]:
            assert abs(np.mean([float(r[raw]) for r in selected])-result[aggregate])<1e-10
    audit=dict(passed=True,source_files_hash_verified=hashes,unique_scenes=len(dev|test),
               development_scenes=len(dev),evaluation_scenes=len(test),complete_comparisons=len(rows),
               cached_predictions_recomputed=len(rows),maximum_absolute_score_discrepancy=errors,
               official_split_ids_verified=len(images),official_split_train_count=795,official_split_test_count=654,
               configuration_sha256=config['configuration_sha256'],
               explanation='Independent recomputation from saved numerical depth arrays and original reference depth; exact paired coverage and source hashes verified.')
    (ROOT/'results/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
