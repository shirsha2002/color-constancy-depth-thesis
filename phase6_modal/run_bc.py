r"""Steps B and C: 10,000-epoch PCCC on Modal T4 using the frozen phase-5 trainer.

Run:  .venv\Scripts\python.exe phase6_modal/make_inputs.py   (once, local)
      .venv-modal\Scripts\modal.exe run --detach phase6_modal/run_bc.py
Geometry is fixed by each condition's manifest (phase6_modal/inputs/<condition>),
so jobs always use the trainer's native-input path: '<head>_real' in a job
directory name means "input as stored in that manifest", not native geometry.
Outputs: Modal volume 'pccc-results', bc_t4_v1/<condition>/<head>_real_fold<k>/.
"""
from pathlib import Path
import modal

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/root/proj'
RAW = 'phase3/data/pccc/raw'
RUN = 'bc_t4_v1'
B = ['ETH3D_real', 'ETH3D_flat', 'NYU_DIODE_real', 'NYU_DIODE_flat']
C = ['DepthAWB_noise10', 'DepthAWB_noise30', 'DepthAWB_drop50', 'DepthAWB_shuffle']

image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git')
         .pip_install('torch==2.5.1', 'numpy<2.3', 'pandas', 'psutil', 'scipy', 'opencv-python-headless')
         .add_local_dir(LOCAL/'phase4_pccc', f'{REMOTE}/phase4_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'phase5_pccc', f'{REMOTE}/phase5_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'external/Point-Cloud-Color-Constancy', f'{REMOTE}/external/Point-Cloud-Color-Constancy')
         .add_local_dir(LOCAL/'phase6_modal/inputs', f'{REMOTE}/phase6_modal/inputs'))
for d in ['RealAWB_pointcloud/new/pcd16', 'real_awb/label2', 'ETH3D_pointcloud/ETH3D/pcd16', 'ETH3D_label/eth3d_label',
          'NYU_DIODE_pointcloud/QXRE_cpd/cpd16', 'NYU_DIODE_label/all_label']:
    image = image.add_local_dir(LOCAL/RAW/d, f'{REMOTE}/{RAW}/{d}')
volume = modal.Volume.from_name('pccc-results', create_if_missing=True)
app = modal.App('pccc-bc', image=image)


@app.function(gpu='T4', timeout=4 * 3600, volumes={'/out': volume})
def train(condition: str, head: str, fold: int, epochs: int = 10000):
    import argparse, json, os, subprocess, sys
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Files come from a Windows checkout (CRLF); compare them as Windows git does.
    for key, val in [('safe.directory', '*'), ('core.autocrlf', 'true'), ('core.fileMode', 'false')]:
        subprocess.run(['git', 'config', '--global', '--add', key, val], check=True)
    sys.path[:0] = [f'{REMOTE}/phase5_pccc', f'{REMOTE}/phase4_pccc']
    import numpy as np, torch
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in a reused container
    import train as t5
    if not getattr(t5, '_uint32_patched', False):
        # torch 2.5 cannot serialise uint32 tensors; store NumPy RNG keys as int64.
        # Checkpoint bookkeeping only; training numerics are unchanged.
        original = t5.rng_state
        def rng_state(device, order):
            s = original(device, order); s['numpy_keys'] = torch.from_numpy(np.random.get_state()[1].astype(np.int64)); return s
        t5.rng_state = rng_state; t5._uint32_patched = True
    manifest_path = Path(f'{REMOTE}/phase6_modal/inputs/{condition}/manifest.json')
    cfg = t5.config(argparse.Namespace(device='cuda', epochs=epochs, manifest=manifest_path))
    cfg['scope'] = f'{condition}: Modal T4 run of the frozen phase-5 trainer (steps B/C); geometry set by manifest'
    out = Path('/out') / (RUN if epochs == 10000 else f'{RUN}_smoke{epochs}') / condition
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    result = t5.train_job(out, cfg, json.loads(manifest_path.read_text()), f'{head}_real', fold)
    volume.commit()
    return {k: result.get(k) for k in ('status', 'training_seconds', 'author_reference_selected_epoch')}


@app.local_entrypoint()
def main(epochs: int = 10000):
    import json, time
    plan = [(c, h, f) for c in B for h in ['released', 'weighted'] for f in range(3)] + \
           [(c, 'released', f) for c in C for f in range(3)]
    jobs = {f'{c}/{h}/{f}': train.spawn(c, h, f, epochs) for c, h, f in plan}
    print('LAUNCHED', len(jobs), flush=True)
    while jobs:
        for name, call in list(jobs.items()):
            try:
                print('DONE', name, json.dumps(call.get(timeout=0)), flush=True); del jobs[name]
            except TimeoutError:
                pass
            except Exception as e:
                print('FAILED', name, repr(e), flush=True); del jobs[name]
        time.sleep(15)
    print('ALL FINISHED', flush=True)
