r"""Steps B and C analysis (pools the three official test folds per dataset).

Usage: .venv\Scripts\python.exe phase6_modal/analyze_bc.py phase6_modal/results/bc_t4_v1 phase6_modal/results/depthawb_t4_v1
B: real vs flat geometry on ETH3D and NYU+DIODE (both heads).
C: DepthAWB released head trained+tested on degraded depth, compared with step A,
   plus an inference-only check: step-A clean-depth models tested on degraded depth.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'phase6_modal'), str(ROOT/'phase5_pccc'), str(ROOT/'phase4_pccc')]
from analyze_a import stats
from pilot import angles

INPUTS = ROOT/'phase6_modal/inputs'
PAPER = {'ETH3D': {'with depth': 0.88, 'w/o depth': 1.42}, 'NYU_DIODE': {'with depth': 2.20, 'w/o depth': 2.80},
         'DepthAWB': {'with depth': 1.05, 'w/o depth': 2.39}}


def load_manifest(cond):
    if cond.startswith('DepthAWB_') and cond not in ('DepthAWB_real', 'DepthAWB_flat'):
        return json.loads((INPUTS/cond/'manifest.json').read_text())
    if cond.startswith('DepthAWB'):
        return json.loads((ROOT/'phase5_pccc/manifest.json').read_text())
    return json.loads((INPUTS/cond/'manifest.json').read_text())


def errors_for(job_root, manifest, sel):
    labels = {r['id']: np.load(ROOT/r['label']) for r in manifest['samples']}; out = {}; worst = 0.
    for f in range(3):
        p = json.loads((job_root(f)/f'{sel}_predictions.json').read_text())
        assert p['ids'] == manifest['folds'][str(f)]['test_ids']
        e = angles(p['prediction'], [labels[i] for i in p['ids']])
        worst = max(worst, float(np.abs(e - np.array(p['angular_error_deg'])).max()))
        out.update(zip(p['ids'], e.tolist()))
    return out, worst


def baselines(manifest):
    v = {r['id']: (np.load(ROOT/r['cloud']), np.load(ROOT/r['label'])) for r in manifest['samples']}; res = {'gray_world': {}, 'training_mean': {}}
    for f in range(3):
        fold = manifest['folds'][str(f)]
        mean = np.mean([v[i][1]/np.linalg.norm(v[i][1]) for i in fold['train_ids']], 0)
        for i in fold['test_ids']:
            res['gray_world'][i] = float(angles(v[i][0][:, 3:].mean(0)[None], v[i][1][None])[0])
            res['training_mean'][i] = float(angles(mean[None], v[i][1][None])[0])
    return res


def paired(a, b, rng):
    ids = sorted(a); d = np.array([a[i] - b[i] for i in ids])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return {'mean_deg': float(d.mean()), 'ci95': [float(x) for x in np.percentile(boot, [2.5, 97.5])], 'first_better_count': int((d < 0).sum()), 'n': len(d)}


def inference_only(stepa, rng_unused=None):
    """Step-A clean-depth released_real fixed_final models, scored on degraded test clouds."""
    from model import load, make, forward
    author, _, _ = load(); res = {}
    for cond in ['DepthAWB_noise10', 'DepthAWB_noise30', 'DepthAWB_drop50', 'DepthAWB_shuffle']:
        m = load_manifest(cond); clouds = {r['id']: r for r in m['samples']}; out = {}
        for f in range(3):
            net = make(author, 'released', 'cpu')
            ck = torch.load(stepa/f'released_real_fold{f}/checkpoints/fixed_final.pt', map_location='cpu', weights_only=True)
            net.load_state_dict(ck['state_dict']); net.eval(); ids = m['folds'][str(f)]['test_ids']
            x = torch.from_numpy(np.stack([np.load(ROOT/clouds[i]['cloud']).astype(np.float32).T for i in ids]))
            with torch.inference_mode():
                pred = forward(net, x, 'released').numpy()
            out.update(zip(ids, angles(pred, [np.load(ROOT/clouds[i]['label']) for i in ids]).tolist()))
        res[cond] = out
    return res


def main(bc, stepa):
    bc, stepa = Path(bc), Path(stepa); rng = np.random.default_rng(20261110)
    rows, errs, check = [], {}, 0.
    def add(dataset, cond, method, sel, d):
        rows.append({'dataset': dataset, 'condition': cond, 'method': method, 'selection': sel,
                     **{k: round(float(x), 4) for k, x in stats(list(d.values())).items()}}); errs[f'{cond}:{method}:{sel}'] = d
    for cond, heads in [('ETH3D_real', 2), ('ETH3D_flat', 2), ('NYU_DIODE_real', 2), ('NYU_DIODE_flat', 2),
                        ('DepthAWB_noise10', 1), ('DepthAWB_noise30', 1), ('DepthAWB_drop50', 1), ('DepthAWB_shuffle', 1)]:
        m = load_manifest(cond)
        for head in ['released', 'weighted'][:heads]:
            for sel in ['fixed_final', 'author_test_selected']:
                d, w = errors_for(lambda f: bc/cond/f'{head}_real_fold{f}', m, sel); check = max(check, w)
                add(m['dataset'], cond, f'PCCC {head}', sel, d)
    for geo in ['real', 'flat']:  # step A, released head, for direct comparison
        m = load_manifest('DepthAWB_real')
        for sel in ['fixed_final', 'author_test_selected']:
            d, w = errors_for(lambda f: stepa/f'released_{geo}_fold{f}', m, sel); check = max(check, w)
            add('DepthAWB', f'DepthAWB_{geo}', 'PCCC released', sel, d)
    for cond in ['ETH3D_real', 'NYU_DIODE_real', 'DepthAWB_real']:
        m = load_manifest(cond)
        ds = cond.rsplit('_', 1)[0]
        for name, d in baselines(m).items():
            add(ds, ds, name, 'baseline', d)
    for cond, d in inference_only(stepa).items():
        add('DepthAWB', cond, 'PCCC released, trained on clean depth', 'inference_only_fixed_final', d)
    comps = {}
    for ds in ['ETH3D', 'NYU_DIODE']:
        for head in ['released', 'weighted']:
            for sel in ['fixed_final', 'author_test_selected']:
                comps[f'{ds} {head} {sel}: real-minus-flat'] = paired(errs[f'{ds}_real:PCCC {head}:{sel}'], errs[f'{ds}_flat:PCCC {head}:{sel}'], rng)
    for cond in ['DepthAWB_noise10', 'DepthAWB_noise30', 'DepthAWB_drop50', 'DepthAWB_shuffle']:
        for sel in ['fixed_final', 'author_test_selected']:
            comps[f'{cond} {sel}: degraded-minus-real'] = paired(errs[f'{cond}:PCCC released:{sel}'], errs[f'DepthAWB_real:PCCC released:{sel}'], rng)
            comps[f'{cond} {sel}: degraded-minus-flat'] = paired(errs[f'{cond}:PCCC released:{sel}'], errs[f'DepthAWB_flat:PCCC released:{sel}'], rng)
        comps[f'{cond} inference-only: degraded-minus-clean'] = paired(
            errs[f'{cond}:PCCC released, trained on clean depth:inference_only_fixed_final'], errs['DepthAWB_real:PCCC released:fixed_final'], rng)
    with (bc/'summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (bc/'summary.json').write_text(json.dumps({'rows': rows, 'paper_reference_means': PAPER, 'max_recompute_discrepancy_deg': check}, indent=1))
    (bc/'paired.json').write_text(json.dumps(comps, indent=1))
    (bc/'per_capture_errors.json').write_text(json.dumps(errs, indent=1))
    for r in rows:
        print(f"{r['dataset']:10s} {r['condition']:18s} {r['method']:40s} {r['selection']:27s} mean {r['mean']:.3f} med {r['median']:.3f}")
    for k, v in comps.items():
        print(f"{k:60s} {v['mean_deg']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] first better {v['first_better_count']}/{v['n']}")
    print('max recompute discrepancy', check)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
