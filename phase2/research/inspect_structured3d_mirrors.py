"""Inspect public mirror metadata only, without downloading licensed scene data."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
IDS=['KevinHuang/Structured3D','Gen3DF/Structured3D','yujia0913/structured3d_origin',
     'yanglab/structured3d_remaining','drozdgk/structured3d','RayYoh/structured3d_gaussiancross',
     'Rain729/Structured3D','helioom/3dvlm-structured3d_subset']
def inspect(repo):
    url='https://huggingface.co/api/datasets/'+repo
    try:
        r=requests.get(url,timeout=(10,30));r.raise_for_status();d=r.json()
        names=[v['rfilename'] for v in d.get('siblings',[])]
        out={'repo':repo,'status':r.status_code,'gated':d.get('gated'),'private':d.get('private'),
            'license':d.get('cardData',{}).get('license'),'sha':d.get('sha'),'files':names}
    except requests.RequestException as e:out={'repo':repo,'error':str(e)}
    print(json.dumps({k:v for k,v in out.items() if k!='files'}|{'file_count':len(out.get('files',[])),
        'archive_examples':[n for n in out.get('files',[]) if n.endswith(('.zip','.tar','.zst'))][:10]},ensure_ascii=True),flush=True)
    return out
with ThreadPoolExecutor(max_workers=4) as pool:data=list(pool.map(inspect,IDS))
(ROOT/'structured3d_mirror_metadata.json').write_text(json.dumps(data,indent=2),encoding='utf8')
