r"""Modal timing benchmark for the frozen PCCC training step (no results are kept).

Run:  .venv-modal\Scripts\modal.exe run phase6_modal/bench.py
Trains weighted_real fold0 for a few epochs on several machine types and reports
seconds per epoch, so the full study can be priced before it is launched.
"""
from pathlib import Path
import modal

LOCAL = Path(__file__).resolve().parents[1]
REMOTE = '/root/proj'
DATA = 'phase3/data/pccc/raw'

image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git')
         .pip_install('torch==2.5.1', 'numpy<2.3', 'pandas', 'psutil', 'scipy', 'opencv-python-headless')
         .add_local_dir(LOCAL/'phase4_pccc', f'{REMOTE}/phase4_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'phase5_pccc', f'{REMOTE}/phase5_pccc', ignore=['results/**'])
         .add_local_dir(LOCAL/'external/Point-Cloud-Color-Constancy', f'{REMOTE}/external/Point-Cloud-Color-Constancy')
         .add_local_dir(LOCAL/DATA/'RealAWB_pointcloud/new/pcd16', f'{REMOTE}/{DATA}/RealAWB_pointcloud/new/pcd16')
         .add_local_dir(LOCAL/DATA/'real_awb/label2', f'{REMOTE}/{DATA}/real_awb/label2'))
app = modal.App('pccc-bench', image=image)


def run(device, epochs, threads):
    import json, os, random, sys, time
    import numpy as np, torch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import subprocess
    # Files come from a Windows checkout (CRLF); compare them as Windows git does.
    for key, val in [('safe.directory', '*'), ('core.autocrlf', 'true'), ('core.fileMode', 'false')]:
        subprocess.run(['git', 'config', '--global', '--add', key, val], check=True)
    sys.path[:0] = [f'{REMOTE}/phase5_pccc', f'{REMOTE}/phase4_pccc']
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    from model import load, make, forward
    import pilot as old
    m = json.loads(Path(f'{REMOTE}/phase5_pccc/manifest.json').read_text())
    values = {r['id']: (np.load(f"{REMOTE}/{r['cloud']}").astype(np.float32),
                        np.load(f"{REMOTE}/{r['label']}").astype(np.float32)) for r in m['samples']}
    ids = m['folds']['0']['train_ids']; test = m['folds']['0']['test_ids']
    author, data, utils = load(); rot = data.PcdColor.__new__(data.PcdColor)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    net = make(author, 'weighted', device); opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    order = np.random.default_rng(1); times = []
    for epoch in range(epochs):
        t = time.perf_counter(); net.train()
        perm = [ids[i] for i in order.permutation(len(ids))]
        for k in range(0, len(perm), 16):
            x, y = old.tensor_batch(perm[k:k+16], values, 'real', True, rot, .15)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); loss = utils.get_angular_loss(forward(net, x, 'weighted'), y)
            loss.backward(); opt.step()
        net.eval()
        with torch.inference_mode():
            for k in range(0, len(test), 16):
                x, _ = old.tensor_batch(test[k:k+16], values, 'real'); forward(net, x.to(device), 'weighted')
        if device == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        print(f'{device} epoch {epoch+1}/{epochs}: {times[-1]:.2f}s', flush=True)
    steady = float(np.median(times[5:]))
    return {'device': device, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
            'threads': threads, 'epochs': epochs, 'sec_per_epoch_median': steady,
            'hours_per_10k_epochs': steady * 10000 / 3600, 'final_train_loss': float(loss)}


@app.function(gpu='T4', timeout=1200)
def t4(epochs: int): return run('cuda', epochs, 4)

@app.function(gpu='L4', timeout=1200)
def l4(epochs: int): return run('cuda', epochs, 4)

@app.function(gpu='A10G', timeout=1200)
def a10g(epochs: int): return run('cuda', epochs, 4)

@app.function(cpu=4, memory=4096, timeout=1200)
def cpu4(epochs: int): return run('cpu', epochs, 4)

@app.function(cpu=2, memory=2048, timeout=1200)
def cpu2(epochs: int): return run('cpu', epochs, 2)


@app.local_entrypoint()
def main(epochs: int = 40):
    import json
    import time
    calls = {name: f.spawn(epochs) for name, f in
             [('t4', t4), ('l4', l4), ('a10g', a10g), ('cpu4', cpu4), ('cpu2', cpu2)]}
    while calls:
        for name, c in list(calls.items()):
            try:
                print(name, json.dumps(c.get(timeout=0)), flush=True); del calls[name]
            except TimeoutError:
                pass
            except Exception as e:
                print(name, 'FAILED', repr(e), flush=True); del calls[name]
        time.sleep(5)
