"""Probe only the official public Vari sharing links, without credentials."""
from __future__ import annotations
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
SHARE='https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87'
token='u!'+base64.urlsafe_b64encode(SHARE.encode()).decode().rstrip('=')
routes={
  'official_share':SHARE,
  'official_share_download_parameter':SHARE+'&download=1',
  'public_shares_metadata':'https://api.onedrive.com/v1.0/shares/'+token+'/root',
  'public_shares_content':'https://api.onedrive.com/v1.0/shares/'+token+'/root/content',
}
report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source':SHARE,'authentication':'none','checks':[]}
for label,url in routes.items():
    item={'route':label,'url':url}
    tick=time.perf_counter()
    try:
        with requests.get(url,stream=True,allow_redirects=True,timeout=(15,30)) as response:
            item.update(status=response.status_code,final_url=response.url,
                content_type=response.headers.get('content-type'),
                content_length=response.headers.get('content-length'),
                content_disposition=response.headers.get('content-disposition'),
                redirects=[{'status':r.status_code,'url':r.url,'location':r.headers.get('location')} for r in response.history])
            prefix=next(response.iter_content(32768),b'')
            if 'text' in (item['content_type'] or '') or 'json' in (item['content_type'] or ''):
                text=prefix.decode(response.encoding or 'utf8',errors='replace')
                (ROOT/(label+'_response.txt')).write_text(text,encoding='utf8')
                item['saved_response_prefix']=label+'_response.txt'
                item['response_prefix']=text[:500]
            else:
                item['binary_prefix_hex']=prefix[:32].hex()
                item['bytes_sampled']=len(prefix)
    except Exception as exc:
        item.update(error_type=type(exc).__name__,error=str(exc))
    item['elapsed_seconds']=round(time.perf_counter()-tick,3)
    report['checks'].append(item)
    print(json.dumps(item,ensure_ascii=True),flush=True)
    (ROOT/'access_probe.json').write_text(json.dumps(report,indent=2),encoding='utf8')
