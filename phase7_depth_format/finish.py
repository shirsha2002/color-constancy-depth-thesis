"""Finite local completion worker: monitor, download, audit and report; no training launch.

Uses the existing Modal client and only the already-running declared app/volume.
Does not publish or stage unrelated files. Does not restart failed training.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/'phase7_depth_format'
OUT=HOME/'results/nyu_frame_v1'
CLIENT=ROOT/'.venv-modal/Scripts/modal.exe'
PYTHON=ROOT/'.venv/Scripts/python.exe'
APP='ap-dVtAGklMO3lKAygdynLAK9'
DEADLINE=datetime(2026,11,16,2,35,4,tzinfo=timezone.utc)+timedelta(seconds=3000)


def write_status(state,**kw):
    value={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),
           'approved_usd':8,**kw}
    path=HOME/'completion_status.json'
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(value),flush=True)


def run(*args):
    subprocess.run([str(a) for a in args],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if args.dry_run:
        print(json.dumps({'mode':'dry-run','cloud_training_launch':False,'approved_existing_app':APP,
                          'download_target':str(OUT),'audit_script':str(HOME/'analyze.py'),
                          'old_phase5_restart':False,'automatic_training_retry':False},indent=2));return
    os.environ['PYTHONUTF8']='1';os.environ['PYTHONIOENCODING']='utf-8'
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while True:
            launch=json.loads((HOME/'launch.json').read_text())
            jobs=list(launch['jobs'].values())
            post_save=[j for j in jobs if j['status']=='failed' and
                       "dict() got multiple values for keyword argument 'fold'" in j.get('error','')]
            failed=[j for j in jobs if j['status']=='failed' and j not in post_save]
            if failed:raise RuntimeError(f'Cloud training failure: {failed}; no retry')
            complete=sum(j['status']=='complete' for j in jobs)
            write_status('waiting_for_cloud',complete_jobs=complete,total_jobs=len(jobs),
                         saved_result_return_errors_requiring_audit=len(post_save))
            # The v1 wrapper raises only after train_job saves all artifacts and
            # volume.commit returns. Do not rerun paid training. Download after
            # all calls end, then require nine complete records plus full audit.
            if len(jobs)==9 and complete+len(post_save)==9:break
            if datetime.now(timezone.utc)>DEADLINE:
                # Global safeguard: stop this owned app rather than leave inputs
                # queued/retried indefinitely. This allocates no new compute.
                run(CLIENT,'app','stop','-y',APP)
                raise TimeoutError('Completion deadline exceeded; owned cloud app stopped. No retry.')
            time.sleep(30)
        write_status('downloading_complete_results')
        OUT.parent.mkdir(parents=True,exist_ok=True)
        assert not OUT.exists(),'Refuse to overwrite a local result tree'
        # Modal preserves the remote basename inside an existing destination.
        # Pass the existing parent, so the result becomes exactly OUT.
        run(CLIENT,'volume','get','pccc-results','nyu_frame_v1',OUT.parent)
        if not (OUT/'camera_real').exists():
            raise RuntimeError('Unexpected download layout; manual review required, no move/delete')
        write_status('independent_audit')
        run(PYTHON,HOME/'analyze.py')
        run(PYTHON,HOME/'plot_results.py')
        audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='PASS'
        smoke=json.loads((HOME/'smoke.json').read_text())
        pairs=json.loads((OUT/'paired.json').read_text())
        results=json.loads((OUT/'summary.json').read_text())
        rows={r['condition']:r for r in results if r['selection']=='fixed_final' and r['group']=='all'}
        message=("Professor, I tested the NYU-v2 + DIODE coordinate-convention hypothesis using nine new PCCC models "
                 "(10,000 epochs, three official folds), reusing the earlier native baselines. "
                 f"Mean final-epoch angular errors were {rows['native']['mean']:.2f} degrees for native geometry, "
                 f"{rows['camera_real']['mean']:.2f} for consistent camera axes, "
                 f"{rows['camera_unitdepth']['mean']:.2f} after per-scene median-depth normalization, "
                 f"and {rows['camera_flat']['mean']:.2f} for the matched canonical flat control. "
                 "RGB and illumination labels were unchanged. All declared comparisons and the independent prediction/checkpoint audit "
                 "are recorded in phase7_depth_format/RESULTS.md. This is a representation diagnostic, not the proposed reliability-aware fusion. "
                 "One seed per fold and prior development exposure limit the conclusions.\n")
        (HOME/'QQ_Update_English.txt').write_text(message,encoding='utf-8',newline='\n')
        status=("# Camera-frame and scale study: completed and audited\n\n"
                "16 November 2026. Nine new released-head PCCC models completed10000epochs on the official NYU/DIODE folds. "
                "Read RESULTS.md and results/nyu_frame_v1/audit.json for all measured results and limits. "
                "No reliability gate has been implemented.\n\n"
                f"Training time: {audit['training_gpu_hours']:.3f} GPU-hours. "
                f"Training compute planning allowance (GPU+maximum CPU/RAM): ${audit['training_resource_planning_usd_upper']:.2f}; "
                "smoke/startup/build overhead is additional. This is not an invoice. User-approved cap was$8. "
                "No extra training or automatic retry is pending. The cancelled laptop study remains stopped.\n\n"
                "An initial short timing run returned a TorchVersion object incompatible with the lightweight client; "
                "it was changed to a plain string and the timing test repeated. This overhead is covered by the reserve.\n")
        (HOME/'STATUS.md').write_text(status,encoding='utf-8',newline='\n')
        write_status('complete_and_audited',complete_jobs=9,audit=str(OUT/'audit.json'),
                     training_gpu_hours=audit['training_gpu_hours'])
    except BaseException as exc:
        write_status('needs_review',error=repr(exc),automatic_retry=False)
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()

