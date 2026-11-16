"""Read existing volume progress; never starts a compute Function."""
from pathlib import Path
import json
import modal

HOME=Path(__file__).resolve().parent
CONDITIONS=['camera_real','camera_unitdepth','camera_flat']


def main():
    volume=modal.Volume.from_name('pccc-results',create_if_missing=False)
    jobs=[]
    for condition in CONDITIONS:
        for fold in range(3):
            path=f'nyu_frame_v1/{condition}/released_real_fold{fold}/progress.json'
            try:
                doc=json.loads(b''.join(volume.read_file(path)))
                row={k:doc.get(k) for k in ['status','epoch','target_epochs','elapsed_seconds','remaining_seconds_estimate','updated_utc']}
            except FileNotFoundError:
                row={'status':'not_yet_visible_in_volume'}
            jobs.append(dict(condition=condition,fold=fold,**row))
    report={'approved_usd':8,'jobs':jobs,
            'note':'Volume progress may lag. This status command does not launch GPU work. No test error is inspected.'}
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':main()

