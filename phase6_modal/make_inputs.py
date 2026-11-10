r"""Build frozen manifests (and derived clouds) for steps B and C.

Run:  .venv\Scripts\python.exe phase6_modal/make_inputs.py
B: ETH3D and NYU+DIODE, native ('real') and ray-preserving flat geometry.
C: DepthAWB with degraded depth (fixed per-capture corruption, same for train and test).
RGB and labels are never changed. Missing geometry stays XYZ=0 as in the release.
"""
import csv, hashlib, json, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PCCC = 'phase3/data/pccc'
OUT = ROOT/'phase6_modal/inputs'
DATASETS = {
    'ETH3D': ('archives/ETH3d_folds.csv', 'raw/ETH3D_pointcloud/ETH3D/pcd16', 'raw/ETH3D_label/eth3d_label'),
    'NYU_DIODE': ('archives/NYU-v2_DIODE_folds.csv', 'raw/NYU_DIODE_pointcloud/QXRE_cpd/cpd16', 'raw/NYU_DIODE_label/all_label'),
    'DepthAWB': ('archives/Depth_AWB_folds.csv', 'raw/RealAWB_pointcloud/new/pcd16', 'raw/real_awb/label2'),
}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def valid(x): return np.abs(x[:, 2]) > 1e-8


def flat(x, rng):
    v = valid(x); x[v, :3] /= np.abs(x[v, 2:3]); return x


def noise(sigma):
    def f(x, rng):
        v = valid(x); x[v, :3] *= (1 + sigma * rng.standard_normal(v.sum()))[:, None]; return x
    return f


def drop(frac):
    def f(x, rng):
        idx = np.flatnonzero(valid(x)); x[rng.choice(idx, int(round(frac * len(idx))), replace=False), :3] = 0; return x
    return f


def shuffle(x, rng):
    x[:, :3] = x[rng.permutation(len(x)), :3]; return x


CONDITIONS = {  # name: (dataset, transform or None for native)
    'ETH3D_real': ('ETH3D', None), 'ETH3D_flat': ('ETH3D', flat),
    'NYU_DIODE_real': ('NYU_DIODE', None), 'NYU_DIODE_flat': ('NYU_DIODE', flat),
    'DepthAWB_noise10': ('DepthAWB', noise(.10)), 'DepthAWB_noise30': ('DepthAWB', noise(.30)),
    'DepthAWB_drop50': ('DepthAWB', drop(.50)), 'DepthAWB_shuffle': ('DepthAWB', shuffle),
}


def build(name, dataset, transform):
    folds_csv, clouds, labels = DATASETS[dataset]
    rows = [(r['fn'].split('.')[0], int(r['foldnum'])) for r in csv.DictReader(open(ROOT/PCCC/folds_csv))]
    samples = []
    for sid, fold in rows:
        src = ROOT/PCCC/clouds/f'{sid}.npy'; lab = ROOT/PCCC/labels/f'{sid}.npy'
        assert src.exists() and lab.exists(), sid
        cloud = src
        if transform is not None:
            x = np.load(src).astype(np.float64)
            rng = np.random.default_rng([20261110, zlib.crc32(f'{name}/{sid}'.encode())])
            y = transform(x.copy(), rng)
            assert np.array_equal(y[:, 3:], x[:, 3:]), 'RGB must be unchanged'
            cloud = OUT/name/'clouds'/f'{sid}.npy'; cloud.parent.mkdir(parents=True, exist_ok=True); np.save(cloud, y)
        samples.append({'id': sid, 'official_fold': fold, 'cloud': cloud.relative_to(ROOT).as_posix(), 'cloud_sha256': sha(cloud),
                        'label': lab.relative_to(ROOT).as_posix(), 'label_sha256': sha(lab)})
    folds = {str(f): {'train_ids': [s['id'] for s in samples if s['official_fold'] != f],
                      'test_ids': [s['id'] for s in samples if s['official_fold'] == f]} for f in range(3)}
    manifest = {'condition': name, 'dataset': dataset, 'transform': None if transform is None else name.split('_')[-1],
                'source_folds_csv': f'{PCCC}/{folds_csv}', 'points_per_cloud': 256, 'samples': samples, 'folds': folds}
    path = OUT/name/'manifest.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding='utf-8', newline='\n')
    print(name, len(samples), {f: len(v['test_ids']) for f, v in folds.items()})


if __name__ == '__main__':
    for n, (d, t) in CONDITIONS.items():
        build(n, d, t)
