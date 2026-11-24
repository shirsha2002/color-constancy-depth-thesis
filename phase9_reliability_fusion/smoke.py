"""Local implementation checks on training captures only; no accuracy claim."""
from pathlib import Path
import json
import numpy as np
import torch
from data import ROOT, MANIFESTS, load, split, damage, DAMAGE, sha
from model import MODES, ReliabilityFusion, make_model, objective, reliability_cues

HOME = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261116)
    torch.use_deterministic_algorithms(True)
    manifests = {}; inventories = {}
    for dataset in MANIFESTS:
        m, values = load(dataset)
        manifests[dataset] = (m, values)
        inventories[dataset] = dict(samples=len(values), folds={str(f):{k:len(v) for k,v in split(m,f).items()} for f in range(3)})
    manifest, values = manifests['ETH3D']
    ids = split(manifest, 0)['train'][:4]
    x = torch.from_numpy(np.stack([values[i][0].T for i in ids]))
    label = torch.from_numpy(np.stack([values[i][1] for i in ids]))
    rng = np.random.default_rng(20261116)
    net = ReliabilityFusion('learned').eval()
    original = net(x)
    assert original['illuminant'].shape == (4,3) and torch.isfinite(original['illuminant']).all()
    assert (original['illuminant']>0).all()
    assert ((original['gate']>=0)&(original['gate']<=1)).all()
    invalid = x[:, 2:3].abs() <= 1e-8
    assert (original['gate'][invalid] == 0).all()
    altered_rgb = x.clone(); altered_rgb[:, 3:] *= .7
    torch.testing.assert_close(original['gate'], net(altered_rgb)['gate'], rtol=0, atol=0)
    permutation = torch.randperm(256)
    torch.testing.assert_close(original['illuminant'], net(x[:,:,permutation])['illuminant'], rtol=1e-5, atol=1e-6)
    missing = x.clone(); missing[:,:3] = 0
    rgb_model = ReliabilityFusion('rgb'); rgb_model.load_state_dict(net.state_dict())
    torch.testing.assert_close(net(missing)['illuminant'], rgb_model(missing)['illuminant'], rtol=0, atol=0)
    assert (net(missing)['gate']==0).all()
    singleton = missing.clone(); singleton[:,:3,0] = torch.tensor([.1,.2,1.])
    cues, _ = reliability_cues(singleton[:,:3])
    assert torch.isfinite(cues).all()
    rejected = False
    bad = x.clone(); bad[0,0,0] = float('nan')
    try: net(bad)
    except ValueError: rejected = True
    assert rejected
    corruption_checks = []
    for kind in DAMAGE:
        a = values[ids[0]][0]
        b = damage(a, kind, np.random.default_rng(99))
        np.testing.assert_array_equal(b[:,3:], a[:,3:])
        np.testing.assert_array_equal(b, damage(a, kind, np.random.default_rng(99)))
        if kind.startswith('noise'):
            v = np.abs(a[:,2])>1e-8
            np.testing.assert_allclose(a[v,:2]/a[v,2:3], b[v,:2]/b[v,2:3], rtol=2e-6, atol=2e-6)
            np.testing.assert_array_equal(np.sign(a[v,2]), np.sign(b[v,2]))
        out = net(torch.from_numpy(b.T[None]))
        assert torch.isfinite(out['illuminant']).all()
        corruption_checks.append(kind)
    gradients = {}; capacities = {}
    damaged = torch.from_numpy(np.stack([damage(values[i][0], 'noise30', rng).T for i in ids]))
    for mode in (*MODES, 'pccc'):
        torch.manual_seed(20261116)
        model = make_model(mode).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        loss = objective(model(x), model(damaged), label)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if mode == 'learned':
            for name in ('color','geometry','gate','head'):
                norm = sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)
                assert norm > 0, f'No gradient in {name}'
                gradients[name] = norm
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        capacities[mode] = model.capacity()
    # Check save/load produces the same answer on this implementation fixture.
    checkpoint = HOME/'smoke_checkpoint.pt'
    torch.save(net.state_dict(), checkpoint)
    restored = ReliabilityFusion('learned').eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(net(x)['illuminant'], restored(x)['illuminant'], rtol=0, atol=0)
    result = dict(status='PASS_IMPLEMENTATION_CHECKS', scope='Not an accuracy study', inventories=inventories,
                  tested_training_capture_ids=ids, tested_dataset='ETH3D', tested_outer_fold=0,
                  forward_finite=True, missing_geometry_rgb_fallback_exact=True,
                  permutation_invariance_checked=True, gate_excludes_rgb_labels_and_corruption_metadata=True,
                  nonfinite_input_rejected=True, deterministic_corruptions_checked=corruption_checks,
                  one_optimizer_step_per_mode=True, learned_branch_gradient_l1=gradients,
                  save_load_exact=True, capacity=capacities,
                  source_sha256={n:sha(HOME/n) for n in ['model.py','data.py','train.py','smoke.py','DESIGN.md']},
                  manifest_sha256={d:sha(ROOT/p) for d,p in MANIFESTS.items()}, paid_compute_launched=False)
    (HOME/'implementation_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
