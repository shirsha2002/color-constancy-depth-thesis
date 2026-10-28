"""Download and inspect the official anonymously shared Vari test ZIP."""
import hashlib
import json
import time
import zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RESEARCH=Path(__file__).resolve().parent
DEST=RESEARCH.parent/'data/vari'
DEST.mkdir(parents=True,exist_ok=True)
URL='https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87&download=1'
target=DEST/'test_dataset.zip'
partial=DEST/'test_dataset.zip.partial'
limit=2_000_000_000
if not target.exists():
    tick=time.perf_counter();digest=hashlib.sha256();done=0;last=0
    session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,connect=2,status=3,
        backoff_factor=2,status_forcelist=[429,500,502,503,504],respect_retry_after_header=True)))
    print('Requesting official public ZIP, with at most3 transient-error retries.',flush=True)
    with session.get(URL,stream=True,timeout=(20,90)) as response:
        response.raise_for_status()
        length=int(response.headers.get('content-length',0))
        if length>limit:raise ValueError('Download exceeds2GBsafetycap')
        meta={'source_url':URL,'final_url':response.url,'content_length':length,
              'content_type':response.headers.get('content-type'),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'authentication':'none; public share link with standard download parameter'}
        print(json.dumps(meta),flush=True)
        with partial.open('wb') as stream:
            for block in response.iter_content(1024*1024):
                if not block:continue
                done+=len(block)
                if done>limit:raise ValueError('Download exceeds2GBsafetycap')
                digest.update(block);stream.write(block)
                if done-last>=32*1024*1024:
                    print(f'{done/1024**2:.1f}MiB/{length/1024**2:.1f}MiB in{time.perf_counter()-tick:.0f}s',flush=True);last=done
    if length and done!=length:raise ValueError(f'Incomplete download{done}/{length}')
    if not zipfile.is_zipfile(partial):raise ValueError('ResponseisnotZIP')
    partial.replace(target)
    meta.update(bytes=done,sha256=digest.hexdigest(),elapsed_seconds=time.perf_counter()-tick)
    (DEST/'download_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
with zipfile.ZipFile(target) as archive:
    files=archive.infolist()
    report={'archive':str(target),'members':len(files),'compressed_bytes':target.stat().st_size,
       'uncompressed_bytes':sum(f.file_size for f in files),
       'extensions':dict(Counter(Path(f.filename).suffix.lower() for f in files if not f.is_dir())),
       'sample_names':[f.filename for f in files[:50]]}
    (DEST/'archive_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
