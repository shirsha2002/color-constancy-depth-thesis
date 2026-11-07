"""Download the small official PCCC illumination labels, folds and point clouds.

Full-resolution 1.5/2.5GB image archives are deliberately not fetched here.
Point clouds are CC training assets, not substitutes for full aligned depth.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,zipfile
from pathlib import Path
import acquire_real_data as common

OUT=common.ROOT/'phase3/data/pccc'
ASSETS=[
 ('Depth_AWB_folds.csv','108tO2hvoLg3sWOIjQxk0FbtmsNANV9oV',1990),
 ('ETH3d_folds.csv','1MvIn9e0nED8dyUhsl4Xexdh0NTAnnOvQ',3132),
 ('NYU-v2_DIODE_folds.csv','1WuZT3BRpuCb-g2x-SeOAXSrJyE3EVgPd',5504),
 ('ETH3D_label.zip','1tBENK8sYEbFSHVEuWLc7U3xiVYDRWYTX',50687),
 ('NYU_DIODE_label.7z','1Fq6lsp-qDDnSBY2oj0ym84euZHVUYgNE',8912),
 ('real_awb.zip','1js23TdY0q7idQGUTqdP5SB0xWRgYevAm',45835),
 ('ETH3D_pointcloud.zip','1XYmoKeRdtIcRQUNJhYhyc4rD50_HqdS7',54494852),
 ('NYU_DIODE_pointcloud.7z','1RCsPCmYUSQqJvU2hru87a-MzVzIEqXDs',95454673),
 ('RealAWB_pointcloud.zip','17SEPj5mN6xh6skiABL_UrxRuyldbUna8',22859533),
]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--labels-only',action='store_true');parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
 frozen_path=OUT/'acquisition_manifest.json'
 frozen=json.loads(frozen_path.read_text(encoding='utf-8')) if frozen_path.exists() else None
 if frozen and common.verify_frozen(frozen,require_complete=args.audit_only):
  print(json.dumps({'frozen_snapshot_audit':'PASS','assets':len(frozen['assets']),'manifest_unchanged':True}),flush=True);return
 if args.audit_only:raise FileNotFoundError('No complete frozen acquisition snapshot')
 common.OUT=OUT;selected=[a for a in ASSETS if not args.labels_only or 'pointcloud' not in a[0].lower()]
 with ThreadPoolExecutor(max_workers=3) as pool:assets=[f.result() for f in as_completed([pool.submit(common.acquire,x) for x in selected])]
 extracted=[]
 for asset in assets:
  path=common.ROOT/asset['path'];destination=(OUT/'raw'/path.stem).resolve();destination.mkdir(parents=True,exist_ok=True)
  if path.suffix=='.zip':
   with zipfile.ZipFile(path) as archive:
    for info in archive.infolist():
     target=(destination/info.filename.replace('\\','/')).resolve()
     if not target.is_relative_to(destination):raise ValueError('Unsafe ZIP path')
     if info.is_dir():continue
     payload=archive.read(info);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
  elif path.suffix=='.7z':
   import py7zr
   with py7zr.SevenZipFile(path) as archive:
    for name in archive.getnames():
     if not (destination/name.replace('\\','/')).resolve().is_relative_to(destination):raise ValueError('Unsafe 7z path')
    archive.extractall(path=destination)
  else:continue
  for output in sorted(destination.rglob('*')):
   if output.is_file():extracted.append({'path':output.relative_to(common.ROOT).as_posix(),'bytes':output.stat().st_size,'sha256':common.sha(output),'source_archive':asset['filename']})
 manifest={'dataset':'PCCC official public labels, original cross-validation fold tables and sampled XYZRGB point clouds','source':'https://github.com/xyxingx/Point-Cloud-Color-Constancy','public_folder':'https://drive.google.com/drive/folders/1qBw_bvaxIvduIm2vzrYhEPX9khTm1Bo9','acquired_utc':datetime.now(timezone.utc).isoformat(),'assets':sorted(assets,key=lambda x:x['filename']),'extracted_files':extracted,'warning':'Original dense RGB and depth are NOT included in this acquisition. XYZRGB points must not be misrepresented as dense aligned metric depth.'}
 if frozen:common.verify_frozen(frozen)
 else:frozen_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
 print(json.dumps({'complete':True,'assets':len(assets),'extracted_files':len(extracted)},indent=2),flush=True)

if __name__=='__main__':main()
