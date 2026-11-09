"""Launch the checked long job without a visible console window."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'phase5_pccc'
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','xpu'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reference-only',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    previous=out/'launch.json'
    if previous.exists():
        record=json.loads(previous.read_text())
        try:active=abs(psutil.Process(record['pid']).create_time()-record['process_created'])<1
        except (psutil.NoSuchProcess,KeyError):active=False
        if active:raise RuntimeError(f'Worker already running for{out.name}: PID{record["pid"]}')
    executable=ROOT/('.venv-xpu/Scripts/python.exe' if args.device=='xpu' else '.venv/Scripts/python.exe')
    if args.reference_only:
        command=[str(executable),'-u',str(HOME/'run_reference.py'),'--device',args.device,'--out',str(out)]
    else:command=[str(executable),'-u',str(HOME/'run_suite.py'),'--device',args.device,'--epochs','10000','--out',str(out)]
    with (out/'worker.log').open('a',encoding='utf-8') as log:
        flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=flags,close_fds=True)
    record={'pid':process.pid,'process_created':psutil.Process(process.pid).create_time(),'launched_utc':datetime.now(timezone.utc).isoformat(),'command':command,
        'status':'launched; check progress and worker.log for actual execution','reference_only':args.reference_only}
    (out/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
