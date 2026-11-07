"""Acquire a new, scene-separated NYU sample without changing prior experiments.

The original verified byte-range reader is reused, including its immutable local
block cache. Only the new phase-three output directory receives exported data.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import download_data as official
OUT=ROOT/'phase3/data/nyu_rgbd'
SEED=20261107

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def export(task):
    split,scene,number=task
    rgb=OUT/'rgb'/f'nyu_{number:05d}.png';depth=OUT/'depth'/f'nyu_{number:05d}.npy'
    if not (rgb.exists() and depth.exists()):
        with h5py.File(official.RemoteHDF5(),'r') as source:
            image=source['images'][number-1].transpose(2,1,0).astype(np.uint8)
            gt=source['depths'][number-1].T.astype(np.float32)
        assert image.shape==(480,640,3) and gt.shape==(480,640)
        rgb_tmp=rgb.with_suffix('.partial');depth_tmp=depth.with_suffix('.partial')
        Image.fromarray(image).save(rgb_tmp,format='PNG');rgb_tmp.replace(rgb)
        with depth_tmp.open('wb') as handle:np.save(handle,gt)
        depth_tmp.replace(depth)
    gt=np.load(depth);valid=np.isfinite(gt)&(gt>.1)&(gt<10)
    return {'id':f'nyu_{number:05d}','split':split,'scene':scene,'original_matlab_id':number,
        'official_split':'test' if split=='evaluation' else 'train',
        'rgb':rgb.relative_to(ROOT).as_posix(),'depth':depth.relative_to(ROOT).as_posix(),
        'rgb_sha256':digest(rgb),'depth_sha256':digest(depth),'depth_units':'meters',
        'width':640,'height':480,'valid_fraction':float(valid.mean())}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training',type=int,default=200)
    parser.add_argument('--validation',type=int,default=40);parser.add_argument('--evaluation',type=int,default=128)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for folder in ['rgb','depth']:(OUT/folder).mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('rgb/\ndepth/\n',encoding='utf-8',newline='\n')
    previous=json.loads((ROOT/'data/nyu_pilot/manifest.json').read_text())
    excluded={r['scene'] for r in previous['images']}
    scene_path=official.OUT/'official_scenes.json';split_path=official.OUT/'splits.mat'
    if not split_path.exists():split_path.write_bytes(official.get(official.SPLIT_URL).content)
    if not scene_path.exists():
        with h5py.File(official.RemoteHDF5(),'r') as dataset:
            scenes=[''.join(chr(int(c)) for c in dataset[ref][:].ravel()) for ref in dataset['scenes'][:].ravel()]
        scene_path.write_text(json.dumps(scenes,indent=2),encoding='utf-8',newline='\n')
    scenes=json.loads(scene_path.read_text())
    splits=loadmat(official.OUT/'splits.mat')
    by_official={}
    for name,key in [('train','trainNdxs'),('test','testNdxs')]:
        groups={}
        for n in splits[key].ravel().astype(int):
            if scenes[n-1] not in excluded:groups.setdefault(scenes[n-1],[]).append(int(n))
        by_official[name]=groups
    assert not (set(by_official['train'])&set(by_official['test']))
    assert args.training+args.validation<=len(by_official['train'])
    assert args.evaluation<=len(by_official['test'])
    rng=np.random.default_rng(SEED)
    train_order=rng.permutation(sorted(by_official['train'])).tolist()
    test_order=rng.permutation(sorted(by_official['test'])).tolist()
    chosen={'training':train_order[:args.training],
        'validation':train_order[args.training:args.training+args.validation],
        'evaluation':test_order[:args.evaluation]}
    tasks=[]
    for split,groups in chosen.items():
        origin='test' if split=='evaluation' else 'train'
        tasks += [(split,scene,int(rng.choice(by_official[origin][scene]))) for scene in sorted(groups)]
    plan={'seed':SEED,'source_url':official.MAT_URL,'split_url':official.SPLIT_URL,
        'previous_manifest_sha256':digest(ROOT/'data/nyu_pilot/manifest.json'),
        'excluded_previously_inspected_scenes':sorted(excluded),'selection':tasks,
        'selection_rule':'One randomly selected frame per scene; original official split; exclude all40 previously inspected scenes; split and frame IDs fixed before model results.',
        'counts':{k:len(v) for k,v in chosen.items()},'scene_disjoint':True}
    frozen=OUT/'selection.json'
    if frozen.exists():assert json.loads(frozen.read_text())==json.loads(json.dumps(plan)), 'Existing split differs'
    else:frozen.write_text(json.dumps(plan,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'frozen_selection':str(frozen),'counts':plan['counts'],'excluded_previous_scenes':len(excluded)},indent=2),flush=True)
    if args.plan_only:return
    manifest_path=OUT/'manifest.json'
    snapshot=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if snapshot and all((ROOT/r[field]).exists() for r in snapshot['images'] for field in ['rgb','depth']):
        assert snapshot['selection_sha256']==digest(frozen)
        for r in snapshot['images']:
            for field in ['rgb','depth']:assert digest(ROOT/r[field])==r[field+'_sha256'],f'Changed frozen input: {r[field]}'
        print('Frozen NYU snapshot audit: PASS; 368 RGB/depth pairs; manifest unchanged.',flush=True);return
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for record in executor.map(export,tasks):
            records.append(record)
            if len(records)%16==0:print(f'Verified NYU pairs: {len(records)}/{len(tasks)}',flush=True)
    result={'dataset':'NYU Depth V2 labeled, new scene-separated phase3 subset',
        'homepage':official.HOMEPAGE,'source_url':official.MAT_URL,
        'selection_sha256':digest(frozen),'selection_seed':SEED,'counts':plan['counts'],
        'depth_reference':'Original Kinect-derived depths field; authors projected/inpainted missing values; metric meters.',
        'radiometry':'Original camera-processed sRGB; this real RGB-D dataset does not provide paired physical illuminant changes or calibrated illuminant labels.',
        'prior_scene_overlap':False,'images':records}
    if snapshot:assert snapshot==result,'Reacquired content differs from frozen NYU snapshot'
    else:manifest_path.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(f'Complete: {len(records)} new real RGB-D scenes.',flush=True)

if __name__=='__main__':main()
