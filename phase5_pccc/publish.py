"""Publish audited numeric results through an isolated Git worktree only.

No checkout, staging or mutation of the user's working branch. No raw inputs,
checkpoints, logs or credentials are copied. A non-fast-forward push is refused.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'phase5_pccc'
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.STDOUT)
def sha(payload):return hashlib.sha256(payload).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',action='store_true');p.add_argument('--check-source-only',action='store_true');args=p.parse_args()
    out=args.out.resolve();assert out.is_relative_to(HOME/'results')
    cfg=json.loads((out/'config.json').read_text())
    git('fetch','origin')
    for name,expected in {**cfg['sources'],cfg['manifest_path']:cfg['manifest_sha256']}.items():
        assert sha(git('show','origin/main:'+name))==expected,'Published source differs from audited run: '+name
    if args.check_source_only:
        print(json.dumps({'publication_source_gate':'PASS','configuration_sha256':cfg['sha256']}));return
    audit=json.loads((out/'independent_audit.json').read_text());assert audit['audit']=='PASS'
    assert sha((out/'predictions.csv').read_bytes())==audit['predictions_csv_sha256']
    assert sha((HOME/'analyze.py').read_bytes())==audit['auditor_source_sha256']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work=(HOME/'tmp'/f'publication_{out.name}_{stamp}').resolve();assert work.is_relative_to((HOME/'tmp').resolve())
    git('worktree','add','--detach',str(work),'origin/main')
    relative=out.relative_to(ROOT);destination=work/relative
    if (destination/'config.json').exists():
        previous=json.loads((destination/'config.json').read_text());assert previous['sha256']==cfg['sha256'],'Refuse replacing a different frozen run'
    count=0
    for path in out.rglob('*'):
        if not path.is_file() or 'checkpoints' in path.parts or path.suffix not in ['.json','.csv','.md']:continue
        if path.name in ['launch.json','publication_status.json','progress.json','suite_status.json']:continue
        target=(destination/path.relative_to(out)).resolve();assert target.is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);count+=1
    status=work/'phase5_pccc'/('REFERENCE_STATUS.md' if args.reference else 'STATUS.md')
    status.write_text('# '+('CPU reference complete' if args.reference else 'DepthAWB paper-budget suite complete')+'\n\n'
        +f"Audited {audit['trained_jobs']} trained jobs at {cfg['epochs']} epochs each on **{cfg['device']}**. "
        +f"Verified {audit['prediction_rows']} prediction rows for {audit['heldout_captures']} held-out captures.\n\n"
        +'Fixed-final primary and explicitly test-selected author-reference results are reported separately. '
        +'This covers the stated DepthAWB256point scope, not the entire paper or a new algorithm claim.\n\n'
        +f'[Measured results](results/{out.name}/RESULTS.md) / [independent audit](results/{out.name}/independent_audit.json).\n',encoding='utf-8',newline='\n')
    git('add','--',str(relative).replace('\\','/'),str(status.relative_to(work)).replace('\\','/'),cwd=work)
    git('diff','--cached','--check',cwd=work)
    changed=git('diff','--cached','--name-only',cwd=work).decode().splitlines()
    if not changed:
        print(json.dumps({'publication':'already_current','commit':git('rev-parse','HEAD',cwd=work).decode().strip()}));return
    assert all(n.startswith(str(relative).replace('\\','/')+'/') or n==str(status.relative_to(work)).replace('\\','/') for n in changed)
    git('commit','-m','Publish audited '+('CPU PCCC reference' if args.reference else 'paper-budget DepthAWB reproduction'),cwd=work)
    commit=git('rev-parse','HEAD',cwd=work).decode().strip()
    git('push','origin','HEAD:refs/heads/main',cwd=work)
    remote=git('ls-remote','origin','refs/heads/main',cwd=work).decode().split()[0];assert remote==commit
    (out/'publication_status.json').write_text(json.dumps({'status':'published','github_main_commit':commit,
        'files_copied':count,'user_working_branch_unchanged':True,'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'published':commit,'files':count}),flush=True)

if __name__=='__main__':main()
