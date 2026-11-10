"""Single source of numbers for the phase-6 figures and report.

Reads only saved, recomputed results (analyze_a.py / analyze_bc.py outputs and
the per-job prediction files); nothing here retrains or re-scores a model.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'phase6_modal/results'
A, BC = RES/'depthawb_t4_v1', RES/'bc_t4_v1'
DATASETS = ['ETH3D', 'DepthAWB', 'NYU_DIODE']
NAME = {'ETH3D': 'ETH3D', 'DepthAWB': 'DepthAWB', 'NYU_DIODE': 'NYU-v2 + DIODE'}
DEGRADED = ['DepthAWB_noise10', 'DepthAWB_noise30', 'DepthAWB_drop50', 'DepthAWB_shuffle']
DEGRADED_NAME = {'DepthAWB_noise10': '10% depth noise', 'DepthAWB_noise30': '30% depth noise',
                 'DepthAWB_drop50': '50% of depth removed', 'DepthAWB_shuffle': '3D positions shuffled'}
STATS = ['mean', 'median', 'trimean', 'best25', 'worst25']
# CVPR 2022 paper, Table 2, "Ours (256 Points)" and "Ours (256 Points, w/o depth)".
PAPER = {'ETH3D': {'real': (0.88, 0.53, 0.62, 0.17, 2.15), 'flat': (1.42, 1.02, 1.08, 0.33, 3.11)},
         'DepthAWB': {'real': (1.05, 0.37, 0.55, 0.09, 3.02), 'flat': (2.39, 1.75, 1.90, 0.42, 5.50)},
         'NYU_DIODE': {'real': (2.20, 1.42, 1.58, 0.34, 5.30), 'flat': (2.80, 1.89, 2.22, 0.57, 6.39)}}
T4_USD_PER_HOUR = 0.000164 * 3600  # Modal list price, Nvidia T4

_A = json.loads((A/'summary.json').read_text())['rows']
_BC = json.loads((BC/'summary.json').read_text())['rows']
_PA = json.loads((A/'paired.json').read_text())
_PBC = json.loads((BC/'paired.json').read_text())
ERR = json.loads((BC/'per_capture_errors.json').read_text())


def _one(rows):
    assert len(rows) == 1, rows
    return {k: rows[0][k] for k in STATS + ['n']}


def pccc(dataset, geometry, head='released', sel='fixed_final'):
    """Pooled statistics over the three official test folds."""
    if dataset == 'DepthAWB' and geometry in ('real', 'flat'):
        return _one([r for r in _A if r['method'] == f'{head}_{geometry}' and r['selection'] == sel])
    cond = f'{dataset}_{geometry}' if geometry in ('real', 'flat') else geometry
    return _one([r for r in _BC if r['condition'] == cond and r['method'] == f'PCCC {head}' and r['selection'] == sel])


def clean_trained_on(cond):
    """Step-A clean-depth models (released head, final epoch) scored on a degraded test set."""
    return _one([r for r in _BC if r['condition'] == cond and r['selection'] == 'inference_only_fixed_final'])


def baseline(dataset, name):
    return _one([r for r in _BC if r['dataset'] == dataset and r['selection'] == 'baseline' and r['method'] == name])


def paper(dataset, geometry):
    return dict(zip(STATS, PAPER[dataset][geometry]))


def real_minus_flat(dataset, head='released', sel='fixed_final'):
    if dataset == 'DepthAWB':
        p = _PA[f'{sel}:{head} real-minus-flat']; better = p['real_better_count']
    else:
        p = _PBC[f'{dataset} {head} {sel}: real-minus-flat']; better = p['first_better_count']
    return {'mean': p['mean_deg'], 'lo': p['ci95'][0], 'hi': p['ci95'][1], 'real_better': better, 'n': p['n']}


def degraded_minus(cond, ref='real', sel='fixed_final'):
    key = f'{cond} inference-only: degraded-minus-clean' if ref == 'clean_trained' else f'{cond} {sel}: degraded-minus-{ref}'
    p = _PBC[key]
    return {'mean': p['mean_deg'], 'lo': p['ci95'][0], 'hi': p['ci95'][1], 'degraded_better': p['first_better_count'], 'n': p['n']}


def significant(d):
    return d['lo'] > 0 or d['hi'] < 0


def per_capture(dataset, geometry, head='released', sel='fixed_final'):
    return ERR[f'{dataset}_{geometry}:PCCC {head}:{sel}']


def job_dirs():
    return sorted(A.glob('*_fold?')) + sorted(BC.glob('*/*_fold?'))


def compute_budget():
    jobs = [json.loads((d/'complete.json').read_text()) for d in job_dirs()]
    seconds = sum(j['training_seconds'] for j in jobs)
    last = max(datetime.fromisoformat(j['completed_utc']) for j in jobs)
    return {'models': len(jobs), 'epochs': sorted({j['epochs'] for j in jobs}), 'gpu_hours': seconds/3600,
            'gpu_usd': seconds/3600*T4_USD_PER_HOUR, 'last_completed_shanghai': (last + timedelta(hours=8)).strftime('%d %B %Y')}


def manifest(dataset):
    if dataset == 'DepthAWB':
        return json.loads((ROOT/'phase5_pccc/manifest.json').read_text())
    return json.loads((ROOT/f'phase6_modal/inputs/{dataset}_real/manifest.json').read_text())


def cloud(dataset_or_condition, sid):
    """Return the (256,6) cloud and label used for a scene in a given condition."""
    m = manifest(dataset_or_condition) if dataset_or_condition in DATASETS else \
        json.loads((ROOT/f'phase6_modal/inputs/{dataset_or_condition}/manifest.json').read_text())
    r = next(s for s in m['samples'] if s['id'] == sid)
    return np.load(ROOT/r['cloud']).astype(np.float64), np.load(ROOT/r['label']).astype(np.float64)


def example_scene(dataset):
    """Fixed, error-blind choice: first fold-0 test scene by ID (numeric for DepthAWB)."""
    ids = manifest(dataset)['folds']['0']['test_ids']
    return min(ids, key=int) if dataset == 'DepthAWB' else min(ids)


def missing_depth(dataset, sid):
    r = next(s for s in manifest(dataset)['samples'] if s['id'] == sid)
    return int((np.abs(np.load(ROOT/r['cloud'])[:, 2]) <= 1e-8).sum())


def complete_depth_scene(dataset):
    """Fixed, error-blind choice for depth illustrations: the fold-0 test scene with the
    fewest missing depth points (lowest ID on ties)."""
    ids = sorted(manifest(dataset)['folds']['0']['test_ids'], key=int if dataset == 'DepthAWB' else str)
    return min(ids, key=lambda i: missing_depth(dataset, i))


def prediction(dataset, geometry, sid, head='released', sel='fixed_final'):
    """Saved prediction vector for one scene from the fold that tested it."""
    m = manifest(dataset); fold = next(f for f in '012' if sid in m['folds'][f]['test_ids'])
    root = A/f'{head}_{geometry}_fold{fold}' if dataset == 'DepthAWB' else BC/f'{dataset}_{geometry}'/f'{head}_real_fold{fold}'
    p = json.loads((root/f'{sel}_predictions.json').read_text())
    i = p['ids'].index(sid)
    return np.array(p['prediction'][i]), p['angular_error_deg'][i], int(fold)


def depth_conventions():
    out = {}
    for ds in DATASETS:
        miss = tot = 0; med = {}; sign = {}
        for r in manifest(ds)['samples']:
            a = np.load(ROOT/r['cloud']); v = np.abs(a[:, 2]) > 1e-8
            miss += int((~v).sum()); tot += len(a); z = a[v, 2]
            med[r['id']] = float(np.median(np.abs(z))); sign[r['id']] = int(np.sign(z).mean())
        out[ds] = {'missing_points': miss, 'points': tot, 'median_abs_z': med, 'sign': sign}
    nyu = {k: v for k, v in out['NYU_DIODE']['median_abs_z'].items() if 'door' not in k}
    diode = {k: v for k, v in out['NYU_DIODE']['median_abs_z'].items() if 'door' in k}
    s = out['NYU_DIODE']['sign']
    out['nyu_vs_diode'] = {'nyu_n': len(nyu), 'diode_n': len(diode),
        'nyu_sign': sorted({s[k] for k in nyu}), 'diode_sign': sorted({s[k] for k in diode}),
        'nyu_range': (min(nyu.values()), max(nyu.values())), 'diode_range': (min(diode.values()), max(diode.values())),
        'ratio_of_medians': float(np.median(list(diode.values())) / np.median(list(nyu.values())))}
    return out


if __name__ == '__main__':
    for ds in DATASETS:
        for g in ['real', 'flat']:
            print(ds, g, {s: round(pccc(ds, g, sel=s)['mean'], 3) for s in ['fixed_final', 'author_test_selected']}, 'paper', PAPER[ds][g][0])
        print('  real-minus-flat', {k: round(v, 3) if isinstance(v, float) else v for k, v in real_minus_flat(ds).items()})
    for c in DEGRADED:
        print(c, round(pccc('DepthAWB', c)['mean'], 3), round(clean_trained_on(c)['mean'], 3))
    print(compute_budget())
    dc = depth_conventions(); print(dc['nyu_vs_diode'])
    for ds in DATASETS:
        print(ds, 'missing', dc[ds]['missing_points'], '/', dc[ds]['points'], 'example', example_scene(ds))
