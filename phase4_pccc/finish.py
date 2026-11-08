"""Wait for this frozen pilot's training, then evaluate and audit exactly once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase4_pccc/results/cpu_v1'
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'phase4_pccc'/name),*args],cwd=ROOT,check=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    out=args.out.resolve();cfg=json.loads((out/'config.json').read_text())
    models=[out/'models'/f'{m}_{s}.pt' for m in cfg['variants'] for s in cfg['seeds']]
    while not all(p.exists() for p in models):time.sleep(10)
    if not (out/'evaluation_complete.json').exists():
        run('pilot.py','evaluate','--epochs',str(cfg['epochs']),'--threads',str(cfg['threads']),'--out',str(out))
    run('analyze.py','--out',str(out))
    if out==OUT.resolve():run('build_summary.py')
    print('Pilot evaluation, independent audit, figures and written summary complete. Visual review and publication remain.',flush=True)
if __name__=='__main__':main()
