"""Read public headers/metadata only. No licensed dataset images are fetched."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
URLS={
    'structured3d_panorama00':'https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_00.zip',
    'structured3d_panorama01':'https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_01.zip',
    'structured3d_panorama16':'https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_16.zip',
    'structured3d_perspective00':'https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_00.zip',
    'midepth_figshare':'https://api.figshare.com/v2/articles/24992976',
    'synclight_page':'https://color.cvc.uab.cat/synclight/dataset/',
}
def probe(item):
    key,url=item;out={'name':key,'url':url}
    try:
        if key.endswith('figshare') or key.endswith('page'):
            r=requests.get(url,timeout=(10,30));out.update(status=r.status_code,final_url=r.url,content_type=r.headers.get('content-type'),bytes=len(r.content))
            if r.status_code==200:(ROOT/(key+'.txt')).write_text(r.text,encoding='utf8')
        else:
            r=requests.head(url,allow_redirects=True,timeout=(10,30));out.update(status=r.status_code,final_url=r.url,
                content_type=r.headers.get('content-type'),content_length=r.headers.get('content-length'),accept_ranges=r.headers.get('accept-ranges'))
    except requests.RequestException as e:out.update(error=type(e).__name__,message=str(e))
    print(json.dumps(out),flush=True);return out
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(probe,URLS.items()))
(ROOT/'paired_candidate_access.json').write_text(json.dumps({'checked_at_utc':datetime.now(timezone.utc).isoformat(),'results':results},indent=2),encoding='utf8')
