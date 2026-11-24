"""Prepare ETH3D scale inputs locally; never load a model or contact a cloud API."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
NATIVE = ROOT / 'phase6_modal/inputs/ETH3D_real/manifest.json'
FLAT = ROOT / 'phase6_modal/inputs/ETH3D_flat/manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    payload = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Refusing to overwrite different frozen evidence: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def normalize(x):
    if x.shape != (256, 6) or not np.isfinite(x).all():
        raise ValueError('Expected finite 256-point XYZRGB array')
    valid = np.abs(x[:, 2]) > 1e-8
    if valid.sum() < 16:
        raise ValueError('Insufficient valid geometry')
    scale = float(np.median(np.abs(x[valid, 2])))
    out = x.copy()
    out[valid, :3] /= scale
    return out, scale, valid


def main():
    native = json.loads(NATIVE.read_text())
    flat = json.loads(FLAT.read_text())
    assert native['folds'] == flat['folds']
    assert len(native['samples']) == 195
    flat_records = {r['id']: r for r in flat['samples']}
    records = []
    seen = {}
    subthreshold_nonzero = 0
    subthreshold_captures = 0
    for r in native['samples']:
        for field in ['cloud', 'label']:
            assert sha(ROOT / r[field]) == r[field + '_sha256']
        x = np.load(ROOT / r['cloud'], allow_pickle=False)
        assert x.dtype == np.float64
        y, scale, valid = normalize(x)
        tiny = int(np.any(x[~valid, :3] != 0, axis=1).sum())
        subthreshold_nonzero += tiny
        subthreshold_captures += int(tiny > 0)
        np.testing.assert_array_equal(y[:, 3:], x[:, 3:])
        np.testing.assert_array_equal(y[~valid], x[~valid])
        np.testing.assert_allclose(y[valid, :3] * scale, x[valid, :3], rtol=1e-12, atol=1e-12)
        label = np.load(ROOT / r['label'], allow_pickle=False)
        assert label.shape == (3,) and np.isfinite(label).all()
        fr = flat_records[r['id']]
        assert fr['label_sha256'] == r['label_sha256']
        assert sha(ROOT / fr['cloud']) == fr['cloud_sha256']
        expected_flat = x.copy()
        expected_flat[valid, :3] /= np.abs(x[valid, 2:3])
        np.testing.assert_array_equal(np.load(ROOT / fr['cloud']), expected_flat)
        path = HOME / 'inputs/ETH3D_unitdepth/clouds' / f"{r['id']}.npy"
        if path.exists():
            np.testing.assert_array_equal(np.load(path), y)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, y, allow_pickle=False)
        records.append(dict(r, cloud=path.relative_to(ROOT).as_posix(), cloud_sha256=sha(path),
                            native_cloud=r['cloud'], native_cloud_sha256=r['cloud_sha256'],
                            median_abs_z_divisor=scale, valid_points=int(valid.sum())))
        seen.setdefault(r['cloud_sha256'], []).append(r['id'])
    ids = [r['id'] for r in records]
    assert len(set(ids)) == 195 and set(ids) == set(flat_records)
    heldout = []
    for fold in native['folds'].values():
        assert len(set(fold['train_ids'])) == 130 and len(set(fold['test_ids'])) == 65
        assert not set(fold['train_ids']) & set(fold['test_ids'])
        assert set(fold['train_ids'] + fold['test_ids']) == set(ids)
        heldout.extend(fold['test_ids'])
    assert len(heldout) == len(set(heldout)) == 195
    manifest = dict(condition='ETH3D_unitdepth', dataset='ETH3D', points_per_cloud=256,
                    transform='XYZ/median(abs(valid Z)); native axes; RGB and invalid XYZ unchanged',
                    source_manifest=NATIVE.relative_to(ROOT).as_posix(), source_manifest_sha256=sha(NATIVE),
                    flat_reference_manifest_sha256=sha(FLAT),
                    prepare_source_sha256=sha(Path(__file__)), protocol_sha256=sha(HOME / 'PROTOCOL.md'),
                    samples=records, folds=native['folds'])
    freeze(HOME / 'inputs/ETH3D_unitdepth/manifest.json', manifest)
    preflight = dict(status='PASS_INPUT_PREPARATION_ONLY', samples=195, derived_clouds=195,
                     folds=3, train_per_fold=130, test_per_fold=65,
                     median_abs_z_range=[min(r['median_abs_z_divisor'] for r in records),
                                         max(r['median_abs_z_divisor'] for r in records)],
                     invalid_points=sum(256-r['valid_points'] for r in records),
                     subthreshold_nonzero_xyz_points_preserved=subthreshold_nonzero,
                     captures_with_subthreshold_nonzero_xyz=subthreshold_captures,
                     exact_duplicate_native_cloud_groups=[v for v in seen.values() if len(v)>1],
                     duplicate_scope='Exact file bytes only; does not establish scene independence',
                     protocol_sha256=manifest['protocol_sha256'],
                     manifest_sha256=sha(HOME / 'inputs/ETH3D_unitdepth/manifest.json'),
                     model_training_launched=False, accuracy_results_available=False)
    freeze(HOME / 'preflight.json', preflight)
    print(json.dumps(preflight, indent=2))


if __name__ == '__main__':
    main()

