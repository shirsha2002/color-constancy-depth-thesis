"""Freeze inputs and run one bounded serial local-GPU accuracy pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import zlib
import numpy as np
from data import ROOT, MANIFESTS, DAMAGE, load, damage, sha

HOME=Path(__file__).resolve().parent
EVIDENCE=HOME/'pilot_v1'
RUNS=HOME/'runs/eth3d_pilot_v1'
CONDITIONS={'rgb':('rgb',True,.1),'ordinary':('ordinary',True,.1),
            'mask':('mask',True,.1),'learned':('learned',True,.1),
            'learned_clean':('learned',False,.1),
            'learned_no_consistency':('learned',True,0.),'pccc':('pccc',True,.1)}
SOURCES=['model.py','data.py','train.py','DESIGN.md','PILOT_PROTOCOL.md',
         'run_pilot.py','analyze_pilot.py','check_pccc_xpu.py','pccc_xpu_repeatability.json']


def freeze(path,value):
    content=(json.dumps(value,indent=2)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content: raise RuntimeError(f'Frozen evidence differs: {path}')
    else:path.write_bytes(content)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    manifest,values=load('ETH3D'); records=[]
    for r in manifest['samples']:
        x,label=values[r['id']]
        for kind in DAMAGE:
            seed=[20261117,zlib.crc32(f"ETH3D/{r['id']}/{kind}".encode())]
            y=damage(x,kind,np.random.default_rng(seed))
            path=HOME/'pilot_inputs'/kind/'clouds'/f"{r['id']}.npy"
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():np.testing.assert_array_equal(np.load(path),y)
            else:np.save(path,y)
            records.append(dict(id=r['id'],condition=kind,cloud=path.relative_to(ROOT).as_posix(),
                                cloud_sha256=sha(path),label=r['label'],label_sha256=r['label_sha256']))
    freeze(EVIDENCE/'evaluation_manifest.json',dict(dataset='ETH3D',samples=records,
          source_manifest_sha256=sha(ROOT/MANIFESTS['ETH3D']),protocol_sha256=sha(HOME/'PILOT_PROTOCOL.md')))
    configuration=dict(scope='ETH3D three-fold reduced-budget development pilot',epochs=100,
                       conditions=CONDITIONS,folds=[0,1,2],models=21,device='xpu',
                       seed_rule='20261116+fold',source_sha256={n:sha(HOME/n) for n in SOURCES},
                       evaluation_manifest_sha256=sha(EVIDENCE/'evaluation_manifest.json'),
                       per_job_timeout_seconds=900,suite_timeout_seconds=5400,paid_compute=False)
    freeze(EVIDENCE/'frozen_run.json',configuration)
    if args.prepare_only:
        print('Prepared 975 fixed test clouds and frozen 21-model protocol; no training launched.');return
    executable=ROOT/'.venv-xpu/Scripts/python.exe'
    started=time.monotonic()
    for fold in range(3):
        for name,(mode,damaged,consistency) in CONDITIONS.items():
            assert all(sha(HOME/n)==h for n,h in configuration['source_sha256'].items())
            out=RUNS/name/f'fold{fold}'
            if out.exists():
                if not (out/'complete.json').exists(): raise RuntimeError(f'Incomplete run requires review: {out}')
                complete=json.loads((out/'complete.json').read_text())
                assert complete['epochs']==100 and complete['status']=='complete'
                assert sha(out/'config.json')==complete['config_sha256']
                cfg=json.loads((out/'config.json').read_text())
                assert cfg['mode']==mode and cfg['fold']==fold and cfg['seed']==20261116+fold
                assert cfg['damage_training']==damaged and cfg['consistency']==consistency
                assert all(sha(HOME/n)==h for n,h in cfg['sources_sha256'].items())
                assert all(sha(out/n)==h for n,h in complete['checkpoints_sha256'].items())
                print(f'Verified complete {name}/fold{fold}',flush=True);continue
            remaining=5400-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Suite time limit reached; no retry')
            command=[str(executable),str(HOME/'train.py'),'--dataset','ETH3D','--mode',mode,
                     '--fold',str(fold),'--seed',str(20261116+fold),'--epochs','100',
                     '--consistency',str(consistency),'--device','xpu','--execute-training','--out',str(out)]
            if damaged:command.append('--damage-training')
            print(f'TRAIN {name}/fold{fold} on local Intel GPU',flush=True)
            subprocess.run(command,cwd=ROOT,check=True,timeout=min(900,remaining))
    print('All 21 training jobs complete; independent analysis is next.',flush=True)


if __name__=='__main__':main()
