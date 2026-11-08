"""Wait for this frozen pilot's training, then evaluate and audit exactly once."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase4_pccc/results/cpu_v1'
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'phase4_pccc'/name),*args],cwd=ROOT,check=True)
def main():
    cfg=json.loads((OUT/'config.json').read_text())
    models=[OUT/'models'/f'{m}_{s}.pt' for m in cfg['variants'] for s in cfg['seeds']]
    while not all(p.exists() for p in models):time.sleep(10)
    if not (OUT/'evaluation_complete.json').exists():
        run('pilot.py','evaluate','--epochs',str(cfg['epochs']),'--threads',str(cfg['threads']))
    run('analyze.py');run('build_summary.py')
    print('Pilot evaluation, independent audit, figures and written summary complete. Visual review and publication remain.',flush=True)
if __name__=='__main__':main()
