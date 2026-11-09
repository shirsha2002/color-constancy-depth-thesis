"""Lightweight progress display: no torch import and no experiment mutation."""
import json
from pathlib import Path
import psutil

HOME=Path(__file__).resolve().parent
def main():
    rows=[]
    for path in sorted((HOME/'results').glob('*/config.json')):
        cfg=json.loads(path.read_text());run=path.parent
        found=False
        for status in sorted(run.glob('*/progress.json')):
            try:r=json.loads(status.read_text())
            except (OSError,json.JSONDecodeError):continue
            alive=False
            try:
                command=psutil.Process(r['pid']).cmdline()
                alive='--out' in command and Path(command[command.index('--out')+1]).resolve()==run.resolve()
            except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError,IndexError):pass
            state=r['status'] if r['status']!='running' or alive else 'not_running; inspect worker.log and resume checkpoint'
            found=True;rows.append({'run':run.name,'job':status.parent.name,'device':r['device'],'status':state,'process_alive':alive,
                'epoch':r['epoch'],'target':r['target_epochs'],'elapsed_minutes':round(r['elapsed_seconds']/60,1),
                'remaining_hours_estimate':round(r['remaining_seconds_estimate']/3600,2),'updated_utc':r['updated_utc']})
        if not found:rows.append({'run':run.name,'device':cfg['device'],'status':'initialized; no training progress yet'})
    print(json.dumps({'paper_budget_runs':rows},indent=2))

if __name__=='__main__':main()
