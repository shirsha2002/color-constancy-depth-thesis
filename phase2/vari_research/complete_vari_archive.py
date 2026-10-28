"""Complete the official public VariZIP with resumable ordinary byte ranges."""
import hashlib
import json
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from acquire_vari_subset import PublicZipRanges,DEST,SIZE,URL,safe_member_path

def main():
    tick=time.perf_counter();target=DEST/'test_dataset.zip'
    prefix=DEST/'test_dataset.zip.partial'
    parts=DEST/'zip_parts';parts.mkdir(exist_ok=True)
    if not target.exists():
        prefix_bytes=prefix.stat().st_size if prefix.exists() else 0
        if prefix_bytes>SIZE:raise ValueError('Unexpectedpartialsize')
        chunk=8*1024*1024
        jobs=[(i,start,min(SIZE-1,start+chunk-1)) for i,start in enumerate(range(prefix_bytes,SIZE,chunk))]
        local=threading.local()
        def job(item):
            i,start,end=item;part=parts/f'{i:04d}-{start}-{end}.bin'
            if part.exists() and part.stat().st_size==end-start+1:return part
            if not hasattr(local,'client'):local.client=PublicZipRanges()
            b=local.client.fetch(start,end);temp=part.with_suffix('.partial');temp.write_bytes(b);temp.replace(part)
            return part
        completed=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(job,j):j for j in jobs}
            for i,f in enumerate(as_completed(futures),1):
                f.result();completed.append(futures[f]);
                done=prefix_bytes+sum(end-start+1 for _,start,end in completed)
                print(f'Archive{done/1024**2:.1f}/{SIZE/1024**2:.1f}MiB; {i}/{len(jobs)}ranges; elapsed{time.perf_counter()-tick:.0f}s',flush=True)
        assembling=DEST/'test_dataset.zip.assembling';digest=hashlib.sha256()
        with assembling.open('wb') as out:
            sources=([prefix] if prefix_bytes else [])+[parts/f'{i:04d}-{start}-{end}.bin' for i,start,end in jobs]
            for source in sources:
                with source.open('rb') as src:
                    while block:=src.read(4*1024*1024):out.write(block);digest.update(block)
        if assembling.stat().st_size!=SIZE:raise ValueError('ArchiveSizeMismatch')
        with zipfile.ZipFile(assembling) as z:
            bad=z.testzip()
            if bad:raise ValueError('ArchiveCRCFailed:'+bad)
        assembling.replace(target)
        metadata={'source_url':URL,'bytes':SIZE,'sha256':digest.hexdigest(),
            'download_method':'OrdinaryanonymousHTTPbyteranges; resumed122MiBpublicdownloadprefix',
            'zip_crc_all_members':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds_for_completion':time.perf_counter()-tick}
        (DEST/'download_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    with zipfile.ZipFile(target) as archive:
        info=archive.infolist();total=sum(i.file_size for i in info)
        if total>2_000_000_000:raise ValueError('Unexpectedextractedsizeabove2GB')
        done=0
        for member in info:
            if member.is_dir():continue
            destination=safe_member_path(member.filename)
            if not destination.exists() or destination.stat().st_size!=member.file_size:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as src,destination.open('wb') as out:shutil.copyfileobj(src,out)
            done+=1
        (DEST/'archive_manifest.json').write_text(json.dumps({'archive':str(target),'members':len(info),
            'files':done,'uncompressed_bytes':total,'all_files_extracted_under':str(DEST/'raw'),
            'path_validation':'Everymemberresolvedinsidephase2/data/vari/raw'},indent=2),encoding='utf8')
    print('FULL_ARCHIVE_READY',target,flush=True)
if __name__=='__main__':main()
