"""Training-only CPU/Intel GPU check; no cloud allocation or benchmark scoring."""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import torch
from data import ROOT, MANIFESTS, split, sha, damage
from model import make_model, objective

HOME = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'xpu'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(status='STARTED', device=args.device, torch=str(torch.__version__),
                  hardware=str(torch.xpu.get_device_name(0)) if args.device=='xpu' and torch.xpu.is_available() else 'CPU',
                  training_only=True, heldout_samples_used=False, paid_compute_launched=False,
                  benchmark_source_sha256=sha(Path(__file__)),
                  model_sha256=sha(HOME/'model.py'), data_sha256=sha(HOME/'data.py'), modes=[])
    path = HOME/f'hardware_{args.device}.json'
    def sync():
        if args.device=='xpu': torch.xpu.synchronize()
    try:
        if args.device=='xpu' and not torch.xpu.is_available():
            raise RuntimeError('Intel GPU unavailable; no silent CPU fallback')
        manifest_path = ROOT/MANIFESTS['ETH3D']
        manifest = json.loads(manifest_path.read_text())
        records = {r['id']:r for r in manifest['samples']}
        ids = split(manifest,0)['train'][:16]
        xs=[]; ys=[]; ds=[]; rng=np.random.default_rng(20261116)
        for sid in ids:
            r=records[sid]
            for field in ('cloud','label'): assert sha(ROOT/r[field])==r[field+'_sha256']
            x=np.load(ROOT/r['cloud']).astype(np.float32)
            xs.append(x.T); ys.append(np.load(ROOT/r['label']).astype(np.float32))
            ds.append(damage(x,'noise30',rng).T)
        arrays=[torch.from_numpy(np.stack(v)) for v in (xs,ds,ys)]
        batch=[v.to(args.device) for v in arrays]
        result.update(training_ids=ids, batch_size=len(ids), manifest_sha256=sha(manifest_path),
                      time_scope='Two forward passes, supervised/consistency loss, backward and Adam step; synchronized; two warmups and eight measured steps',
                      epoch_projection='Compute only, ceil(train_count/16)*step time; excludes loading, validation, checkpointing and startup')
        for mode in ('learned','mask','ordinary','rgb','pccc'):
            print(f'Checking {mode} on {args.device}',flush=True)
            torch.manual_seed(20261116)
            reference=make_model(mode).train()
            initial=copy.deepcopy(reference.state_dict())
            reference.zero_grad(set_to_none=True)
            ref_prediction=reference(arrays[0])['illuminant'].detach()
            ref_loss=objective(reference(arrays[0]),reference(arrays[1]),arrays[2])
            ref_loss.backward()
            ref_gradients={n:p.grad.clone() for n,p in reference.named_parameters() if p.grad is not None}
            target=make_model(mode).to(args.device).train(); target.load_state_dict(initial)
            actual_prediction=target(batch[0])['illuminant'].detach().cpu()
            torch.testing.assert_close(actual_prediction,ref_prediction,rtol=2e-3,atol=2e-3)
            target.zero_grad(set_to_none=True)
            actual_loss=objective(target(batch[0]),target(batch[1]),batch[2]); actual_loss.backward()
            assert torch.isfinite(actual_loss)
            for n,p in target.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
                    torch.testing.assert_close(p.grad.cpu(),ref_gradients[n],rtol=2e-2,atol=2e-3)
            optimizer=torch.optim.Adam(target.parameters(),lr=.0003)
            if args.device=='xpu': torch.xpu.reset_peak_memory_stats()
            times=[]
            for step in range(10):
                sync(); started=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss=objective(target(batch[0]),target(batch[1]),batch[2])
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); sync()
                elapsed=time.perf_counter()-started
                if step>=2: times.append(elapsed)
            assert all(torch.isfinite(p).all() for p in target.parameters())
            record=dict(mode=mode, status='PASS', synchronized_median_seconds_per_step=float(np.median(times)),
                        cpu_prediction_max_absolute_difference=float((actual_prediction-ref_prediction).abs().max()),
                        cpu_gradient_tolerance_passed=True, finite_adam_updates=10,
                        training_seconds_per_100_epochs_compute_only=float(np.median(times)*7*100),
                        training_seconds_per_10000_epochs_compute_only=float(np.median(times)*7*10000))
            if args.device=='xpu': record['framework_peak_allocated_mib']=torch.xpu.max_memory_allocated()/1024**2
            result['modes'].append(record)
            print(json.dumps(record),flush=True)
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAILED', error=repr(exc), automatic_cloud_fallback=False)
        raise
    finally:
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(f'Saved {path}',flush=True)


if __name__=='__main__':
    main()
