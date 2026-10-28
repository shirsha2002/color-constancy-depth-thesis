"""Acquire an explicit exploratory Vari split with ordinary public HTTP ranges.

Reads the public ZIP central directory and only the selected original members.
No login, credentials, or access-control circumvention is used.
"""
from __future__ import annotations
import binascii
import hashlib
import io
import json
import re
import struct
import threading
import time
import zlib
import zipfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RESEARCH=Path(__file__).resolve().parent
PROJECT=RESEARCH.parents[1]
DEST=RESEARCH.parent/'data/vari'
URL='https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87&download=1'
SIZE=824263108
LIGHTS=['Environment','Indoor','SunNoon_Indoor','SunNight_Indoor','SunMorning_Indoor',
        'SunNoon_Indoor_Environment','SunNight_Indoor_Environment','SunMorning_Indoor_Environment',
        'SunNoon','SunNight','SunMorning']
PAT=re.compile(r'(?P<scene>\d+)_(?P<camera>c\d+)_(?P<lighting>.+?)(?P<view>\d{4})\.jpg$')

class PublicZipRanges:
    def __init__(self):
        self.session=requests.Session()
        self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,connect=2,status=3,
            backoff_factor=1,status_forcelist=[429,500,502,503,504],respect_retry_after_header=True)))
        self.final_url=None;self.bytes_received=0;self.requests=0
    def fetch(self,start,end):
        for attempt in range(4):
            try:return self._fetch_once(start,end)
            except (requests.exceptions.SSLError,requests.exceptions.ConnectionError,requests.exceptions.Timeout):
                if attempt==3:raise
                self.session.close();self.__init__();time.sleep(2*(attempt+1))
    def _fetch_once(self,start,end):
        url=self.final_url or URL
        r=self.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
        if r.status_code==401 and self.final_url:
            r.close();r=self.session.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
        r.raise_for_status()
        if r.status_code!=206:raise RuntimeError(f'Server ignored byte range: {r.status_code}')
        expected=f'bytes {start}-{end}/{SIZE}'
        if r.headers.get('content-range')!=expected:raise RuntimeError('UnexpectedContentRange')
        self.final_url=r.url;data=r.content;r.close()
        if len(data)!=end-start+1:raise RuntimeError('Truncated byte range')
        self.bytes_received+=len(data);self.requests+=1
        return data

class RemoteReader(io.RawIOBase):
    def __init__(self,client):self.client=client;self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        return self.position
    def read(self,n=-1):
        if n<0:n=SIZE-self.position
        n=min(n,SIZE-self.position)
        if n<=0:return b''
        b=self.client.fetch(self.position,self.position+n-1);self.position+=len(b);return b

def member_bytes(client,entry):
    # The extra64bytes cover the usual local-header extra fields. The exact
    # parsed lengths determine the compressed payload; retry if more is needed.
    start=entry.header_offset
    need=30+len(entry.filename.encode('utf8'))+len(entry.extra)+entry.compress_size+64
    b=client.fetch(start,min(SIZE-1,start+need-1))
    head=struct.unpack('<4s5H3I2H',b[:30])
    if head[0]!=b'PK\x03\x04':raise RuntimeError('InvalidLocalZIPHeader')
    offset=30+head[-2]+head[-1]
    if offset+entry.compress_size>len(b):b=client.fetch(start,start+offset+entry.compress_size-1)
    compressed=b[offset:offset+entry.compress_size]
    if entry.compress_type==zipfile.ZIP_STORED:data=compressed
    elif entry.compress_type==zipfile.ZIP_DEFLATED:data=zlib.decompress(compressed,-15)
    else:raise RuntimeError(f'UnsupportedZIPCompression{entry.compress_type}')
    if len(data)!=entry.file_size or binascii.crc32(data)&0xffffffff!=entry.CRC:
        raise RuntimeError('ZIPMemberCRCOrSizeMismatch')
    return data

def safe_member_path(name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('UnsafeZIPPath')
    resolved=(DEST/'raw'/Path(*rel.parts)).resolve()
    if not resolved.is_relative_to((DEST/'raw').resolve()):raise ValueError('UnsafeZIPPath')
    return resolved

def main():
    DEST.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    client=PublicZipRanges()
    with zipfile.ZipFile(RemoteReader(client)) as z:
        entries={i.filename:i for i in z.infolist() if not i.is_dir()}
    groups=defaultdict(dict)
    for name in entries:
        match=PAT.search(PurePosixPath(name).name)
        if not match or match['lighting'] not in LIGHTS:continue
        d=match.groupdict();key=(d['scene'],d['camera'],d['view'])
        groups[key][d['lighting']]=name
    complete={k:v for k,v in groups.items() if set(v)==set(LIGHTS)}
    index=[]
    for key,rgb in sorted(complete.items()):
        scene,camera,view=key;base=PurePosixPath(next(iter(rgb.values()))).parent
        depth=str(base/f'{scene}_{camera}_CGeometry_ZDepth{view}.tif')
        extra_types=['CGeometry_NormalsGeometry','CShading_Albedo','Diffuse','Reflect']
        extras={kind:str(base/f'{scene}_{camera}_{kind}{view}.jpg') for kind in extra_types}
        index.append({'scene':scene,'camera':camera,'view':view,'view_group':'_'.join(key),
            'rgb':rgb,'depth':depth,'depth_present':depth in entries,
            'auxiliary_maps':{k:v for k,v in extras.items() if v in entries}})
    index_report={'source_url':URL,'archive_bytes':SIZE,'archive_files':len(entries),
        'archive_uncompressed_bytes':sum(e.file_size for e in entries.values()),
        'extensions':dict(Counter(PurePosixPath(n).suffix.lower() for n in entries)),
        'complete_groups':len(index),'groups_per_scene':dict(Counter(g['scene'] for g in index)),
        'lights':LIGHTS,'groups':index}
    (DEST/'complete_groups.json').write_text(json.dumps(index_report,indent=2),encoding='utf8')
    (DEST/'zip_central_directory.json').write_text(json.dumps([{
        'name':e.filename,'bytes':e.file_size,'compressed_bytes':e.compress_size,
        'crc32':f'{e.CRC:08x}','compression':e.compress_type,'header_offset':e.header_offset,
    } for e in entries.values()],indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in index_report.items() if k!='groups'},indent=2),flush=True)
    rng=np.random.default_rng(20261028)
    room212=sorted([g for g in index if g['scene']=='212'],key=lambda g:g['view_group'])
    room485=sorted([g for g in index if g['scene']=='485'],key=lambda g:g['view_group'])
    shuffled212=[room212[i] for i in rng.permutation(len(room212))]
    shuffled485=[room485[i] for i in rng.permutation(len(room485))]
    if len(room212)<24 or len(room485)<24:raise ValueError('Fewer sourcegroups than requested; review split')
    chosen=[]
    for split,selected in [('training',shuffled212[:20]),('validation',shuffled212[20:24]),('evaluation',shuffled485[:24])]:
        chosen.extend([{**g,'split':split} for g in selected])
    # Save the declared split before any learning or depth inference occurs.
    (DEST/'exploratory_split_groups.json').write_text(json.dumps({'seed':20261028,'groups':chosen,
        'warning':'New exploratory split of released officialTEST assets. Not the official Vari training benchmark.'},indent=2),encoding='utf8')
    wanted={g['depth'] for g in chosen}
    wanted.update(n for g in chosen for n in g['rgb'].values())
    jobs=sorted(wanted)
    local=threading.local();lock=threading.Lock();transferred=[]
    def fetch_one(name):
        target=safe_member_path(name)
        if target.exists() and target.stat().st_size==entries[name].file_size:
            if binascii.crc32(target.read_bytes())&0xffffffff==entries[name].CRC:return name,0
        if not hasattr(local,'client'):local.client=PublicZipRanges()
        data=member_bytes(local.client,entries[name]);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        return name,len(data)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch_one,name):name for name in jobs}
        for i,future in enumerate(as_completed(futures),1):
            name,num=future.result();transferred.append(num)
            if i%12==0 or i==len(jobs):print(f'{i}/{len(jobs)} rawfiles; {sum(transferred)/1024**2:.1f}MiB uncompressed; elapsed{time.perf_counter()-tick:.0f}s',flush=True)
    from prepare_vari_manifest import main as prepare_manifest
    prepare_manifest()
    (DEST/'subset_download_provenance.json').write_text(json.dumps({'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_url':URL,'method':'Ordinary anonymous HTTPbyteranges of publicZIP; ZIPCRC32verified for every extractedmember',
        'raw_files':len(jobs),'selected_groups':len(chosen),'raw_uncompressed_bytes':sum(safe_member_path(n).stat().st_size for n in jobs),
        'elapsed_seconds':time.perf_counter()-tick,'official_train_download':'No public training link in officialREADME'},indent=2),encoding='utf8')
    print('MANIFEST_READY',str(DEST/'manifest.json'),flush=True)

if __name__=='__main__':main()
