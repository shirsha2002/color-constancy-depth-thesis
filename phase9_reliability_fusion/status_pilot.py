"""Read local pilot progress without training, GPU allocation or test scores."""
from pathlib import Path
import json

HOME=Path(__file__).resolve().parent


def main():
    path=HOME/'pilot_v1/frozen_run.json'
    if not path.exists():
        print('No frozen accuracy pilot is prepared.');return
    cfg=json.loads(path.read_text()); rows=[]
    for fold in cfg['folds']:
        for method in cfg['conditions']:
            folder=HOME/'runs/eth3d_pilot_v1'/method/f'fold{fold}'
            epoch=0; state='not_started'
            if (folder/'complete.json').exists():
                state='complete';epoch=100
            elif folder.exists():
                state='started_or_needs_review'
                if (folder/'history.json').exists():
                    try:
                        history=json.loads((folder/'history.json').read_text())
                        epoch=history[-1]['epoch'] if history else 0
                    except (json.JSONDecodeError,PermissionError):state='history_write_in_progress'
            rows.append(dict(method=method,fold=fold,epoch=epoch,state=state))
    audit=HOME/'pilot_v1/audit.json'
    print(json.dumps(dict(complete=sum(r['state']=='complete' for r in rows),total=21,
                          epochs_written=sum(r['epoch'] for r in rows),total_epochs=2100,
                          audit=json.loads(audit.read_text())['status'] if audit.exists() else 'not_complete',
                          jobs=rows),indent=2))


if __name__=='__main__':main()
