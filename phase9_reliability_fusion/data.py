"""Hash-checked released captures and explicit synthetic depth damage."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {'DepthAWB': 'phase5_pccc/manifest.json',
             'ETH3D': 'phase6_modal/inputs/ETH3D_real/manifest.json',
             'NYU_DIODE': 'phase6_modal/inputs/NYU_DIODE_real/manifest.json'}
DAMAGE = ('clean', 'noise10', 'noise30', 'drop50', 'shuffle')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(dataset):
    path = ROOT / MANIFESTS[dataset]
    manifest = json.loads(path.read_text())
    values = {}
    for r in manifest['samples']:
        for field in ('cloud', 'label'):
            if sha(ROOT / r[field]) != r[field+'_sha256']:
                raise ValueError(f'Input hash changed: {r[field]}')
        x, label = np.load(ROOT/r['cloud']).astype(np.float32), np.load(ROOT/r['label']).astype(np.float32)
        if x.shape != (256, 6) or label.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(label).all() or np.linalg.norm(label)==0:
            raise ValueError(f'Invalid sample {r["id"]}')
        values[r['id']] = (x, label)
    return manifest, values


def split(manifest, fold):
    outer = manifest['folds'][str(fold)]
    ordered = sorted(outer['train_ids'], key=lambda sid: hashlib.sha256(f'20261116/{sid}'.encode()).hexdigest())
    count = max(1, round(.2*len(ordered)))
    result = dict(train=ordered[count:], validation=ordered[:count], test=list(outer['test_ids']))
    assert not set(result['train']) & set(result['validation'])
    assert not set(result['train']+result['validation']) & set(result['test'])
    assert set(sum(result.values(), [])) == set(values_id['id'] for values_id in manifest['samples'])
    return result


def damage(x, kind, rng):
    if kind not in DAMAGE:
        raise ValueError(kind)
    out = x.copy()
    valid = np.abs(x[:, 2]) > 1e-8
    if kind in ('noise10', 'noise30'):
        sigma = .1 if kind == 'noise10' else .3
        # Positive multiplicative depth noise; preserve rays and depth sign.
        gain = np.exp(sigma*rng.standard_normal(valid.sum())-.5*sigma*sigma)
        out[valid, :3] *= gain[:, None]
    elif kind == 'drop50':
        ids = np.flatnonzero(valid)
        out[rng.choice(ids, round(len(ids)*.5), replace=False), :3] = 0
    elif kind == 'shuffle':
        out[:, :3] = x[rng.permutation(len(x)), :3]
    assert np.array_equal(out[:, 3:], x[:, 3:])
    return out


def batch(ids, values, rng=None, damage_training=False):
    xs, ys, altered = [], [], []
    for sid in ids:
        x, y = values[sid]
        xs.append(x.T.copy()); ys.append(y)
        kind = DAMAGE[int(rng.integers(len(DAMAGE)))] if damage_training else 'clean'
        altered.append(damage(x, kind, rng).T)
    return np.stack(xs), np.stack(altered), np.stack(ys)
