"""Verify frozen sources and checkpoint identity before final evaluation.

This adds a guard without modifying the frozen numerical runners. It prevents
reusing saved predictions after replacing a trained checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',choices=['global','spatial','controls']);args=parser.parse_args()
    folder=ROOT/'phase3/results'/({'global':'study_v1','spatial':'spatial_v1','controls':'spatial_controls'}[args.run])
    cfg=json.loads((folder/'config.json').read_text())
    runtime=json.loads((ROOT/'phase3/results/runtime.json').read_text())
    assert sha(ROOT/'phase2/mono/cc_model.py')==runtime['frozen_loader_sha256'],'Frozen loader changed'
    assert sha(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth')==cfg['checkpoint_sha256'],'Base checkpoint changed'
    for relative,expected in {**cfg['sources'],**cfg['manifests']}.items():
        assert sha(ROOT/relative)==expected,f'Frozen source/manifest changed: {relative}'
    methods=list(cfg['variant_mode_equivariance_weight'])
    import torch
    hashes={}
    for method in methods:
        path=folder/'models'/f'{method}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['configuration_sha256']==cfg['configuration_sha256']
        hashes[method]=sha(path)
    identity={'configuration_sha256':cfg['configuration_sha256'],'checkpoint_sha256':hashes,
        'frozen_base_checkpoint_sha256':cfg['checkpoint_sha256']}
    path=folder/'evaluation_sources.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Checkpoint changed; refusing stale prediction reuse'
    else:path.write_text(json.dumps(identity,indent=2),encoding='utf-8',newline='\n')
    print(f'Verified {len(methods)} frozen trained checkpoints; starting final evaluation.',flush=True)
    script={'global':'study.py','spatial':'spatial_study.py','controls':'spatial_controls.py'}[args.run]
    subprocess.run([sys.executable,str(ROOT/'phase3/learning'/script),'evaluate',
        '--input-size',str(cfg['input_size']),'--threads',str(cfg['threads']),
        '--steps',str(cfg['training_steps']),'--out',str(folder)],check=True,cwd=ROOT)

if __name__=='__main__':main()
