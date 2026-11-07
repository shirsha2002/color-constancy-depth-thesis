"""Acquire author-linked public ReMID assets without credentials or full-disk scans.

The five files are download shards, not train/validation/test partitions.
Every transfer is size/SHA256 checked and every ZIP member is CRC checked.
Raw data remain outside Git; manifests contain public IDs and file hashes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'phase3/data/remid'
SOURCE = 'https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9#Fn1'
FOLDER = 'https://drive.google.com/drive/folders/1-3txE3KiFiscACxs1Xch8UO1Ax82VBNg'
ASSETS = [
 ('README.md','1LOni6f87S6AGYMJ9RCppClqEwXTRcapa',223),
 ('split_1.zip','1JiInWjnF9cS3_0ieoBBJuFdJCPhfxSIv',241817925),
 ('split_2.zip','1I99cIkvgFa6m33hFuYhkmlmXeG2oPcwg',297421096),
 ('split_3.zip','1CFiXL0j6mkDN7IwVBwS9EF1oiN-laOZS',133585271),
 ('split_4.zip','1YrFKzZI2_yWMzgNbEO1Jfb6UJl3BzSRS',212073769),
 ('split_5.zip','1QTxNVwLOwnNXRHuwJBS1drb2cKM7RCY5',145906305),
]

def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):d.update(chunk)
 return d.hexdigest()

def acquire(asset):
 name,public_id,expected=asset
 target=OUT/'archives'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if not (target.exists() and target.stat().st_size==expected):
  part=target.with_suffix(target.suffix+'.partial')
  for attempt in range(4):
   start=part.stat().st_size if part.exists() else 0
   headers={'Range':f'bytes={start}-'} if start else {}
   try:
    with requests.get('https://drive.usercontent.google.com/download',params={'id':public_id,'export':'download','confirm':'t'},headers=headers,stream=True,timeout=(30,120)) as response:
     response.raise_for_status()
     if 'text/html' in response.headers.get('Content-Type',''):
      raise RuntimeError('Public download returned HTML rather than an asset')
     if start and response.status_code==206:
      if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
       raise RuntimeError('Incorrect resumed byte range')
      mode='ab'
     else:mode='wb'
     with part.open(mode) as f:
      for chunk in response.iter_content(1024*1024):
       if chunk:f.write(chunk)
    if part.stat().st_size!=expected:raise RuntimeError(f'Incorrect size {part.stat().st_size} != {expected}')
    part.replace(target);break
   except (requests.RequestException,RuntimeError):
    if attempt==3:raise
    time.sleep(2**attempt)
 result={'filename':name,'public_drive_id':public_id,'bytes':expected,'sha256':sha(target),'path':target.relative_to(ROOT).as_posix()}
 if name.endswith('.zip'):
  with zipfile.ZipFile(target) as archive:
   result['archive_members']=len(archive.infolist())
   result['archive_uncompressed_bytes']=sum(i.file_size for i in archive.infolist())
   result['member_preview']=[i.filename for i in archive.infolist()[:16]]
 print(json.dumps({'downloaded':result},ensure_ascii=True),flush=True)
 return result

def extract(assets):
 raw=(OUT/'raw').resolve();raw.mkdir(parents=True,exist_ok=True);records=[]
 for item in assets:
  if not item['filename'].endswith('.zip'):continue
  archive_path=ROOT/item['path']
  with zipfile.ZipFile(archive_path) as archive:
   for entry in archive.infolist():
    # Reject paths outside the intended extraction root, including absolute paths.
    name=entry.filename.replace('\\','/')
    target=(raw/name).resolve()
    if not target.is_relative_to(raw):raise ValueError('Unsafe archive member path')
    if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
    payload=archive.read(entry)  # ZipFile verifies member CRC32 while reading.
    if len(payload)!=entry.file_size:raise ValueError('Truncated ZIP member')
    target.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(payload).hexdigest()
    if target.exists() and sha(target)!=digest:raise ValueError('Conflicting members across transport shards')
    if not target.exists():target.write_bytes(payload)
    records.append({'path':target.relative_to(ROOT).as_posix(),'archive':item['filename'],'member':entry.filename,'bytes':entry.file_size,'crc32':f'{entry.CRC:08x}','sha256':digest})
 return records

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--no-extract',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 results=[]
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  for future in as_completed([pool.submit(acquire,x) for x in ASSETS]):results.append(future.result())
 results.sort(key=lambda x:x['filename'])
 manifest={'dataset':'ReMID, author-published release linked by LumiDepth ECCV2026','source_paper':SOURCE,'public_folder':FOLDER,'acquired_utc':datetime.now(timezone.utc).isoformat(),'notice':'split_1..split_5 are transport shards, NOT experiment splits. Dense depth units and sensor alignment must be established separately before use.','assets':results}
 if not args.no_extract:manifest['extracted_files']=extract(results)
 (OUT/'acquisition_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
 print(json.dumps({'complete':True,'archives':len(results),'extracted_files':len(manifest.get('extracted_files',[])),'manifest':str(OUT/'acquisition_manifest.json')},indent=2),flush=True)

if __name__=='__main__':main()
