"""Reacquire only the ~23MB DepthAWB assets needed by this pilot.

Use original immutable acquisition identities; do not rewrite phase-three data
manifests or download the unrelated NYU/DIODE/ETH3D archives.
"""
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase3/data_tools'))
import acquire_real_data as common

def main():
    home=ROOT/'phase3/data/pccc';common.OUT=home
    source=json.loads((home/'acquisition_manifest.json').read_text())
    selected=[a for a in source['assets'] if a['filename'] in ['Depth_AWB_folds.csv','real_awb.zip','RealAWB_pointcloud.zip']]
    assert len(selected)==3
    for asset in selected:
        actual=common.acquire((asset['filename'],asset['public_drive_id'],asset['bytes']))
        assert actual['sha256']==asset['sha256'],'Asset differs from frozen author release'
        archive=ROOT/asset['path']
        if archive.suffix!='.zip':continue
        destination=(home/'raw'/archive.stem).resolve();destination.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target=(destination/member.filename.replace('\\','/')).resolve()
                assert target.is_relative_to(destination),'Unsafe archive path'
                if member.is_dir():continue
                payload=handle.read(member)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert target.read_bytes()==payload,'Existing native file differs'
                else:target.write_bytes(payload)
    print(json.dumps({'DepthAWB_inputs':'PASS','source_assets':3,'source_bytes':sum(a['bytes'] for a in selected)}))

if __name__=='__main__':main()
