"""Read saved results and recompute with scalar acos independently of the analyzer."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import numpy as np

HOME=Path(__file__).resolve().parent
OUT=HOME/'pilot_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='PASS'
    rows=json.loads((OUT/'predictions.json').read_text())
    summaries=json.loads((OUT/'summary.json').read_text())
    per_fold=json.loads((OUT/'per_fold.json').read_text())
    pairs=json.loads((OUT/'paired.json').read_text())
    cfg=json.loads((OUT/'frozen_run.json').read_text())
    assert audit['frozen_run_sha256']==sha(OUT/'frozen_run.json')
    assert audit['analyzer_sha256']==sha(HOME/audit['analyzer_file'])
    assert audit['original_frozen_analyzer_sha256']==sha(HOME/'analyze_pilot.py')
    assert all(sha(HOME/n)==h for n,h in cfg['source_sha256'].items())
    assert len(rows)==13650
    groups=defaultdict(dict); folds=defaultdict(list); max_error=0.
    for r in rows:
        a,b=r['prediction'],r['label']
        dot=sum(float(x)*float(y) for x,y in zip(a,b))
        norm=math.sqrt(sum(float(x)**2 for x in a)*sum(float(x)**2 for x in b))
        assert norm>0 and math.isfinite(norm)
        angle=math.degrees(math.acos(max(-.999999,min(.999999,dot/norm))))
        max_error=max(max_error,abs(angle-r['error_deg']))
        assert abs(angle-r['error_deg'])<1e-8
        key=(r['method'],r['selection'],r['condition'])
        assert r['id'] not in groups[key]
        groups[key][r['id']]=angle
        folds[key+(r['fold'],)].append(angle)
    assert len(groups)==70
    for r in summaries:
        g=groups[(r['method'],r['selection'],r['condition'])]
        assert len(g)==r['n']==195
        x=sorted(g.values()); mean=sum(x)/195; median=x[97]; worst=sum(x[-48:])/48
        for calculated,stored in [(mean,r['mean']),(median,r['median']),(worst,r['worst25'])]:assert abs(calculated-stored)<1e-8
    for r in per_fold:
        x=folds[(r['method'],r['selection'],r['condition'],r['fold'])]
        assert len(x)==r['n']==65 and abs(sum(x)/65-r['mean'])<1e-8
    source=json.loads((OUT/'evaluation_manifest.json').read_text())
    ids=list(dict.fromkeys(r['id'] for r in source['samples']))
    rng=np.random.default_rng(20261118)
    for pair in pairs:
        conditions=['clean'] if pair['endpoint']=='clean' else ['noise10','noise30','drop50','shuffle']
        def average(method,sid):return sum(groups[(method,'fixed_final',c)][sid] for c in conditions)/len(conditions)
        delta=np.array([average('learned',i)-average('mask',i) for i in ids])
        assert abs(float(delta.mean())-pair['mean_deg'])<1e-8
        boot=np.array([delta[rng.integers(0,195,195)].mean() for _ in range(10000)])
        np.testing.assert_allclose(np.percentile(boot,[2.5,97.5]),pair['conditional_capture_bootstrap95'],rtol=0,atol=1e-8)
    for job in audit['jobs']:
        folder=HOME/'runs/eth3d_pilot_v1'/job['method']/f"fold{job['fold']}"
        assert sha(folder/'config.json')==job['config_sha256']
        assert all(sha(folder/n)==h for n,h in job['checkpoints_sha256'].items())
    result=dict(status='PASS',saved_angles_checked=13650,summary_rows_checked=len(summaries),
                per_fold_rows_checked=len(per_fold),primary_intervals_reproduced=2,
                maximum_angle_discrepancy_deg=max_error,checkpoint_hashes_checked=42,
                verifier_sha256=sha(Path(__file__)),
                audited_artifact_sha256={n:sha(OUT/n) for n in ['audit.json','predictions.json','summary.json','per_fold.json','paired.json']})
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
