"""Sequential, resumable all-fold suite; never changes model hyperparameters."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'phase5_pccc'
def now():return datetime.now(timezone.utc).isoformat()
def dump(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n');temp.replace(path)
def run(name,*args):subprocess.run([sys.executable,'-u',str(HOME/name),*args],cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','xpu'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10000);p.add_argument('--primary-only',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    variants=['weighted_real','weighted_flat'] if args.primary_only else ['weighted_real','weighted_flat','released_real','released_flat']
    jobs=[f'{variant}:{fold}' for fold in [0,1,2] for variant in variants]
    run('train.py','init','--device',args.device,'--epochs',str(args.epochs),'--out',str(out))
    for index,job in enumerate(jobs):
        dump(out/'suite_status.json',{'status':'running','device':args.device,'job':job,'job_index':index+1,
            'jobs_total':len(jobs),'schedule':jobs,'updated_utc':now(),'scope':'DepthAWB256points3officialfolds',
            'results_complete':False})
        try:run('train.py','train','--device',args.device,'--epochs',str(args.epochs),'--out',str(out),'--job',job)
        except subprocess.CalledProcessError as e:
            dump(out/'suite_status.json',{'status':'failed','failed_job':job,'exit_code':e.returncode,
                'schedule':jobs,'updated_utc':now(),'results_complete':False});raise
    run('analyze.py','--out',str(out),'--variants',','.join(variants))
    dump(out/'suite_status.json',{'status':'complete','device':args.device,'jobs_total':len(jobs),'schedule':jobs,
        'updated_utc':now(),'results_complete':True,'scope':'DepthAWB256points3officialfolds'})
    run('publish.py','--out',str(out))

if __name__=='__main__':main()
