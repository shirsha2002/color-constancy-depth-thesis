"""Use training captures only to verify exact interrupted/resumed execution."""
import copy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','xpu'],default='cpu');args=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=train.ROOT;home=train.HOME;source=json.loads((home/'manifest.json').read_text())
    ids=source['folds']['0']['train_ids'][:24];wanted=set(ids)
    tiny={'purpose':'Restart preflight only; original fold0 training captures only',
        'samples':[r for r in source['samples'] if r['id'] in wanted],
        'folds':{str(f):{'train_ids':ids[:16],'test_ids':ids[16:]} for f in [0,1,2]}}
    path=home/'tmp/restart_manifest.json';train.dump(path,tiny)
    cfg=train.config(SimpleNamespace(manifest=path.resolve(),device=args.device,epochs=4))
    a=home/f'tmp/uninterrupted_{args.device}_identity_pool';b=home/f'tmp/interrupted_{args.device}_identity_pool'
    train.train_job(a,cfg,tiny,'weighted_real',0)
    train.train_job(b,cfg,tiny,'weighted_real',0,stop_after=2)
    train.train_job(b,cfg,tiny,'weighted_real',0)
    left=torch.load(a/'weighted_real_fold0/checkpoints/fixed_final.pt',weights_only=True,map_location='cpu')['state_dict']
    right=torch.load(b/'weighted_real_fold0/checkpoints/fixed_final.pt',weights_only=True,map_location='cpu')['state_dict']
    identical=left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    assert left.keys()==right.keys() and all(torch.allclose(left[k],right[k],rtol=1e-5,atol=1e-6) for k in left),'Resume changed final weights'
    h1=json.loads((a/'weighted_real_fold0/history.json').read_text());h2=json.loads((b/'weighted_real_fold0/history.json').read_text())
    assert len(h1)==len(h2) and all(abs(a[k]-b[k])<1e-4 for a,b in zip(h1,h2) for k in a),'Resume changed loss histories'
    result={'restart_preflight':'PASS','data_used':'24originalfold0trainingcaptures, no finaltestcaptures',
        'uninterrupted_epochs':4,'interrupted_after_epoch':2,'resumed_epochs':2,
        'device':args.device,'all_final_model_tensors_bitwise_identical':identical,'all_training_and_monitor_histories_identical':h1==h2,
        'weights_within_rtol1e-5_atol1e-6':True,'histories_within1e-4deg':True,
        'trainer_sha256':train.sha(home/'train.py')}
    path=home/('restart_preflight.json' if args.device=='cpu' else 'restart_preflight_xpu.json')
    train.dump(path,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
