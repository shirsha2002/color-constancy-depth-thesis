"""Full-budget single-fold CPU reference, with automatic audit and publication."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'phase5_pccc'
def run(name,*args):subprocess.run([sys.executable,'-u',str(HOME/name),*args],cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','xpu'],required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve()
    run('train.py','train','--device',args.device,'--epochs','10000','--out',str(out),'--job','weighted_real:0')
    run('analyze.py','--out',str(out),'--variants','weighted_real','--folds','0')
    run('publish.py','--out',str(out),'--reference')

if __name__=='__main__':main()
