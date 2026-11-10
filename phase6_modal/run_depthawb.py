r"""Step A: full 10,000-epoch PCCC on DepthAWB, 4 variants x 3 official folds, Modal T4.

Run:  .venv-modal\Scripts\modal.exe run phase6_modal/run_depthawb.py
Training logic is the frozen phase5_pccc/train.py (sources hashed into config);
only the device is CUDA. Outputs go to the Modal volume 'pccc-results' under
depthawb_t4_v1/ and are downloaded with download.ps1.
"""
from pathlib import Path
import modal

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/root/proj'
DATA = 'phase3/data/pccc/raw'
RUN = 'depthawb_t4_v1'

image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git')
         .pip_install('torch==2.5.1', 'numpy<2.3', 'pandas', 'psutil', 'scipy', 'opencv-python-headless')
         .add_local_dir(LOCAL/'phase4_pccc', f'{REMOTE}/phase4_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'phase5_pccc', f'{REMOTE}/phase5_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'external/Point-Cloud-Color-Constancy', f'{REMOTE}/external/Point-Cloud-Color-Constancy')
         .add_local_dir(LOCAL/DATA/'RealAWB_pointcloud/new/pcd16', f'{REMOTE}/{DATA}/RealAWB_pointcloud/new/pcd16')
         .add_local_dir(LOCAL/DATA/'real_awb/label2', f'{REMOTE}/{DATA}/real_awb/label2'))
volume = modal.Volume.from_name('pccc-results', create_if_missing=True)
app = modal.App('pccc-depthawb', image=image)


@app.function(gpu='T4', timeout=3 * 3600, volumes={'/out': volume})
def train(variant: str, fold: int, epochs: int = 10000):
    import argparse, json, os, subprocess, sys
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Files come from a Windows checkout (CRLF); compare them as Windows git does.
    for key, val in [('safe.directory', '*'), ('core.autocrlf', 'true'), ('core.fileMode', 'false')]:
        subprocess.run(['git', 'config', '--global', '--add', key, val], check=True)
    sys.path[:0] = [f'{REMOTE}/phase5_pccc', f'{REMOTE}/phase4_pccc']
    import torch
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in a reused container
    import numpy as np
    import train as t5
    if not getattr(t5, '_uint32_patched', False):
        # torch 2.5 cannot serialise uint32 tensors; store NumPy RNG keys as int64.
        # Checkpoint bookkeeping only; training numerics are unchanged.
        original = t5.rng_state
        def rng_state(device, order):
            s = original(device, order); s['numpy_keys'] = torch.from_numpy(np.random.get_state()[1].astype(np.int64)); return s
        t5.rng_state = rng_state; t5._uint32_patched = True
    manifest_path = Path(f'{REMOTE}/phase5_pccc/manifest.json')
    args = argparse.Namespace(device='cuda', epochs=epochs, manifest=manifest_path)
    cfg = t5.config(args)
    cfg['scope'] += '; Modal T4 rerun of the frozen phase-5 protocol (step A)'
    out = Path('/out') / (RUN if epochs == 10000 else f'{RUN}_smoke{epochs}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    manifest = json.loads(manifest_path.read_text())
    result = t5.train_job(out, cfg, manifest, variant, fold)
    volume.commit()
    return {'variant': variant, 'fold': fold, 'gpu': torch.cuda.get_device_name(), **{k: result.get(k) for k in ('status', 'training_seconds', 'author_reference_selected_epoch')}}


@app.local_entrypoint()
def main(epochs: int = 10000):
    import json, time
    jobs = {f'{v}:{f}': train.spawn(v, f, epochs) for v in ['weighted_real', 'weighted_flat', 'released_real', 'released_flat'] for f in [0, 1, 2]}
    while jobs:
        for name, c in list(jobs.items()):
            try:
                print('DONE', name, json.dumps(c.get(timeout=0)), flush=True); del jobs[name]
            except TimeoutError:
                pass
            except Exception as e:
                print('FAILED', name, repr(e), flush=True); del jobs[name]
        time.sleep(15)
    print('ALL FINISHED', flush=True)
