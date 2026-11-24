"""Explicit local training worker. No Modal API calls or automatic job launch.

Requires a new output directory. CLI --dry-run only freezes/checks the setup;
actual training additionally requires --execute-training. Not paper reproduction.
"""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from data import ROOT, MANIFESTS, load, split, batch, sha
from model import MODES, make_model, objective

HOME = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def score(net, ids, values, device):
    vectors, labels = [], []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            x, _, label = batch(ids[start:start+16], values)
            vectors.extend(net(torch.from_numpy(x).to(device))['illuminant'].cpu().numpy().astype(float).tolist())
            labels.extend(label.astype(float).tolist())
    a, b = np.array(vectors), np.array(labels)
    cosine = (a*b).sum(1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
    angles = np.rad2deg(np.arccos(np.clip(cosine, -.999999, .999999)))
    return dict(ids=ids, prediction=vectors, label=labels, angular_error_deg=angles.tolist(), mean_deg=float(angles.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=MANIFESTS, default='ETH3D')
    p.add_argument('--mode', choices=(*MODES, 'pccc'), default='learned')
    p.add_argument('--fold', type=int, choices=range(3), default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--damage-training', action='store_true')
    p.add_argument('--consistency', type=float, default=.1)
    p.add_argument('--device', choices=('cpu','cuda','xpu'), default='cpu')
    p.add_argument('--out', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--execute-training', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.consistency < 0:
        p.error('Require positive epochs and nonnegative consistency weight')
    manifest, values = load(args.dataset)
    membership = split(manifest, args.fold)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    net = make_model(args.mode)
    config = dict(dataset=args.dataset, mode=args.mode, seed=args.seed, fold=args.fold,
                  epochs=args.epochs, batch_size=16, learning_rate=.0003,
                  damage_training=args.damage_training, consistency=args.consistency,
                  device=args.device, torch=str(torch.__version__), numpy=np.__version__,
                  deterministic_algorithms=True, capacity=net.capacity(), splits=membership,
                  manifest=MANIFESTS[args.dataset], manifest_sha256=sha(ROOT/MANIFESTS[args.dataset]),
                  sources_sha256={n:sha(HOME/n) for n in ['model.py','data.py','train.py','DESIGN.md']},
                  known_exposure='Previously inspected benchmark; development evidence; capture splits do not prove room independence',
                  primary_checkpoint='Fixed final epoch', secondary_checkpoint='Best inner-validation mean; earliest tie',
                  training_augmentation='Explicit synthetic depth damage only; no color/XYZ rotation augmentation in this prototype')
    if args.mode == 'pccc':
        import upstream
        config['original_upstream'] = upstream.identities()
        config['original_wrapper_sha256'] = sha(ROOT/'phase5_pccc/model.py')
    if args.dry_run:
        print(json.dumps(config, indent=2)); return
    if not args.execute_training or args.out is None:
        p.error('Training requires --execute-training and a NEW --out directory')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite or silently resume an existing run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    if args.device == 'xpu':
        if not torch.xpu.is_available():
            raise RuntimeError('Intel GPU unavailable; no CPU fallback')
        if args.mode == 'pccc':
            clearance_path = HOME/'pccc_xpu_repeatability.json'
            clearance = json.loads(clearance_path.read_text())
            if (clearance['status'] != 'PASS_SAME_XPU_REPEATABILITY' or
                clearance['model_sha256'] != sha(HOME/'model.py') or
                clearance['checker_sha256'] != sha(HOME/'check_pccc_xpu.py')):
                raise RuntimeError('Original PCCC requires current same-XPU repeatability evidence')
            config['xpu_baseline_repeatability_sha256'] = sha(clearance_path)
            config['cross_backend_gradient_identity'] = 'Not established; earlier failed strict comparison retained'
        torch.xpu.manual_seed_all(args.seed)
    out.mkdir(parents=True)
    write(out/'config.json', config)
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
    # Separate RNGs keep sample orders and damage draws matched across modes.
    order_rng = np.random.default_rng(args.seed)
    damage_rng = np.random.default_rng(args.seed+1)
    history = []; best = float('inf')
    for epoch in range(1, args.epochs+1):
        net.train(); losses = []
        ids = [membership['train'][i] for i in order_rng.permutation(len(membership['train']))]
        for start in range(0, len(ids), 16):
            clean, damaged, labels = batch(ids[start:start+16], values, damage_rng, args.damage_training)
            clean, damaged, labels = [torch.from_numpy(a).to(args.device) for a in (clean, damaged, labels)]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(net(clean), net(damaged), labels, args.consistency)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stopped without retry')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        validation = score(net, membership['validation'], values, args.device)
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), validation_mean_deg=validation['mean_deg']))
        if validation['mean_deg'] < best:
            best = validation['mean_deg']
            torch.save(dict(state_dict=net.state_dict(), epoch=epoch, config=config), out/'validation_selected.pt')
        write(out/'history.json', history)
    torch.save(dict(state_dict=net.state_dict(), epoch=args.epochs, config=config), out/'fixed_final.pt')
    # First outer-test access is after all optimizer updates and selection.
    write(out/'fixed_final_predictions.json', score(net, membership['test'], values, args.device))
    state = torch.load(out/'validation_selected.pt', map_location=args.device, weights_only=True)
    net.load_state_dict(state['state_dict'])
    write(out/'validation_selected_predictions.json', score(net, membership['test'], values, args.device))
    write(out/'complete.json', dict(status='complete', epochs=args.epochs,
          validation_selected_epoch=state['epoch'], config_sha256=sha(out/'config.json'),
          checkpoints_sha256={n:sha(out/n) for n in ['fixed_final.pt','validation_selected.pt']}))
    print(f'Completed explicit local run: {out}')


if __name__ == '__main__':
    main()
