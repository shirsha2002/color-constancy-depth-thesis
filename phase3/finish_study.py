"""Finish frozen evaluations and audits after all source-only training is done.

Safe to resume after an interruption once no other finish process is running.
This coordinator never selects a model from final-test scores.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,check=True)
def ready(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def main():
    controls=ROOT/'phase3/results/spatial_controls'
    cfg=json.loads((controls/'config.json').read_text())
    expected=list(cfg['variant_mode_equivariance_weight'])
    while not all(ready(controls/f'{method}_training.json') for method in expected):time.sleep(15)
    run('phase3/record_selection.py')
    if not ready(controls/'summary.json'):run('phase3/run_verified.py','controls')
    paths=[ROOT/'phase3/results'/folder/'summary.json' for folder in ['study_v1','spatial_v1','spatial_controls','float_probe']]
    while not all(ready(path) for path in paths):time.sleep(15)
    run('phase3/merge_results.py')
    run('phase3/analyze.py')
    run('phase3/build_report.py')
    print('All frozen experiments, independent audits, analysis and draft report complete. Visual PDF QA and GitHub main update remain.',flush=True)

if __name__=='__main__':main()
