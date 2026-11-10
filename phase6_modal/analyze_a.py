r"""Step A analysis: pool the three official DepthAWB test folds (185 captures).

Usage: .venv\Scripts\python.exe phase6_modal/analyze_a.py phase6_modal/results/depthawb_t4_v1
Writes summary.csv / summary.json / paired.json next to the results and
recomputes every angular error from the saved prediction vectors.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'phase4_pccc'))
from pilot import angles

VARIANTS = ['released_real', 'released_flat', 'weighted_real', 'weighted_flat']
PAPER = {'paper 256 points, with depth': (1.05, .37, .55, .09, 3.02),
         'paper 256 points, w/o depth': (2.39, 1.75, 1.90, .42, 5.50)}


def stats(e):
    e = np.sort(np.asarray(e, dtype=np.float64)); q = len(e) // 4
    q1, q2, q3 = np.percentile(e, [25, 50, 75])
    return {'n': len(e), 'mean': e.mean(), 'median': q2, 'trimean': (q1 + 2*q2 + q3) / 4,
            'best25': e[:q].mean(), 'worst25': e[-q:].mean()}


def main(run):
    run = Path(run); manifest = json.loads((ROOT/'phase5_pccc/manifest.json').read_text())
    values = {r['id']: (np.load(ROOT/r['cloud']), np.load(ROOT/r['label'])) for r in manifest['samples']}
    errors, rows, check = {}, [], 0.
    for sel in ['fixed_final', 'author_test_selected']:
        for v in VARIANTS:
            ids, err = [], []
            for f in range(3):
                p = json.loads((run/f'{v}_fold{f}'/f'{sel}_predictions.json').read_text())
                assert p['ids'] == manifest['folds'][str(f)]['test_ids']
                e = angles(p['prediction'], [values[i][1] for i in p['ids']])
                check = max(check, float(np.abs(e - np.array(p['angular_error_deg'])).max()))
                ids += p['ids']; err += e.tolist()
            errors[(sel, v)] = dict(zip(ids, err))
    # Baselines, computed per fold from that fold's training captures only.
    for name in ['gray_world', 'training_mean']:
        d = {}
        for f in range(3):
            fold = manifest['folds'][str(f)]
            mean = np.mean([values[i][1]/np.linalg.norm(values[i][1]) for i in fold['train_ids']], 0)
            for i in fold['test_ids']:
                pred = values[i][0][:, 3:].mean(0) if name == 'gray_world' else mean
                d[i] = float(angles(pred[None], values[i][1][None])[0])
        errors[('baseline', name)] = d
    ids = sorted(errors[('baseline', 'gray_world')], key=int); assert len(ids) == 185
    for (sel, v), d in errors.items():
        rows.append({'selection': sel, 'method': v, **{k: round(float(x), 4) for k, x in stats([d[i] for i in ids]).items()}})
    for name, s in PAPER.items():
        rows.append({'selection': 'published', 'method': name, 'n': 185,
                     **dict(zip(['mean', 'median', 'trimean', 'best25', 'worst25'], s))})
    rng = np.random.default_rng(20261110); paired = {}
    for sel in ['fixed_final', 'author_test_selected']:
        for head in ['released', 'weighted']:
            diff = np.array([errors[(sel, f'{head}_real')][i] - errors[(sel, f'{head}_flat')][i] for i in ids])
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
            paired[f'{sel}:{head} real-minus-flat'] = {'mean_deg': float(diff.mean()),
                'ci95': [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                'real_better_count': int((diff < 0).sum()), 'n': len(diff)}
    with (run/'summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (run/'summary.json').write_text(json.dumps({'rows': rows, 'max_recompute_discrepancy_deg': check}, indent=2))
    (run/'paired.json').write_text(json.dumps(paired, indent=2))
    (run/'per_capture_errors.json').write_text(json.dumps({f'{s}:{v}': d for (s, v), d in errors.items()}, indent=1))
    for r in rows:
        print(f"{r['selection']:22s} {r['method']:30s} mean {r['mean']:.3f} med {r['median']:.3f} W25 {r['worst25']:.3f}")
    print(json.dumps(paired, indent=1)); print('max recompute discrepancy', check)


if __name__ == '__main__':
    main(sys.argv[1])
