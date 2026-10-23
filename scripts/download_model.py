"""Download the official indoor metric-depth checkpoint with a content hash."""
import hashlib
import json
import time
from pathlib import Path
import requests

URL='https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true'
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'

def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if not TARGET.exists():
        tmp=TARGET.with_suffix('.pth.partial')
        for attempt in range(4):
            try:
                with requests.get(URL,stream=True,timeout=(25,90)) as response:
                    response.raise_for_status()
                    total=int(response.headers.get('content-length',0));done=0;last=0
                    with tmp.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk);done+=len(chunk)
                            if time.monotonic()-last>5:
                                print(f'Checkpoint {done/1e6:.1f} MB / {total/1e6:.1f} MB',flush=True);last=time.monotonic()
                if tmp.stat().st_size<90_000_000:
                    raise ValueError('Checkpoint is unexpectedly small')
                tmp.replace(TARGET);break
            except (requests.RequestException,ValueError) as exc:
                print(f'Attempt {attempt+1}: {type(exc).__name__}',flush=True)
                if attempt==3:raise
    digest=hashlib.sha256(TARGET.read_bytes()).hexdigest()
    record=dict(url=URL,filename=TARGET.name,bytes=TARGET.stat().st_size,sha256=digest,
                model='Depth Anything V2 Metric Hypersim Small',encoder='vits',max_depth_m=20)
    TARGET.with_suffix('.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
