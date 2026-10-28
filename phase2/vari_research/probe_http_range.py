"""Check ordinary HTTP byte-range support on the official public ZIP URL."""
import json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'access_probe.json').read_text())
hit=next(c for c in p['checks'] if c['route']=='official_share_download_parameter')
url=hit['url'];size=int(hit['content_length']);start=size-65536
with requests.get(url,headers={'Range':f'bytes={start}-{size-1}'},stream=True,timeout=(20,45)) as r:
    out={'status':r.status_code,'content_range':r.headers.get('content-range'),
         'content_length':r.headers.get('content-length'),'accept_ranges':r.headers.get('accept-ranges')}
    print(json.dumps(out),flush=True)
    if r.status_code==206:
        data=r.content
        (ROOT/'zip_tail.bin').write_bytes(data)
        out.update(bytes_received=len(data),eocd_offset=data.rfind(b'PK\x05\x06'))
    (ROOT/'http_range_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out),flush=True)
