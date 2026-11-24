"""Independent saved-array checks, including trainer float32 conversion."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = HOME / 'inputs/ETH3D_unitdepth/manifest.json'
    m = json.loads(path.read_text())
    assert sha(HOME / 'PROTOCOL.md') == m['protocol_sha256']
    assert sha(HOME / 'prepare.py') == m['prepare_source_sha256']
    native = json.loads((ROOT / m['source_manifest']).read_text())
    assert sha(ROOT / m['source_manifest']) == m['source_manifest_sha256']
    assert m['folds'] == native['folds']
    flat_path = ROOT / 'phase6_modal/inputs/ETH3D_flat/manifest.json'
    assert sha(flat_path) == m['flat_reference_manifest_sha256']
    source = {r['id']: r for r in native['samples']}
    maximum_ray_change = 0.0
    for r in m['samples']:
        original = source[r['id']]
        assert r['official_fold'] == original['official_fold']
        assert r['native_cloud'] == original['cloud']
        assert r['native_cloud_sha256'] == original['cloud_sha256']
        assert r['label'] == original['label'] and r['label_sha256'] == original['label_sha256']
        for field in ['cloud', 'label', 'native_cloud']:
            assert sha(ROOT / r[field]) == r[field + '_sha256']
        a = np.load(ROOT / r['native_cloud'], allow_pickle=False)
        b = np.load(ROOT / r['cloud'], allow_pickle=False)
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape == (256, 6)
        assert np.isfinite(b).all()
        v = np.abs(a[:, 2]) > 1e-8
        assert int(v.sum()) == r['valid_points']
        np.testing.assert_array_equal(v, np.abs(b[:, 2]) > 1e-8)
        np.testing.assert_array_equal(a[:, 3:], b[:, 3:])
        np.testing.assert_array_equal(a[~v, :3], b[~v, :3])
        np.testing.assert_array_equal(np.sign(a[v, 2]), np.sign(b[v, 2]))
        scale = np.median(np.abs(a[v, 2]))
        assert scale == r['median_abs_z_divisor']
        np.testing.assert_allclose(np.median(np.abs(b[v, 2])), 1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(b[v, :3], a[v, :3] / scale, rtol=1e-12, atol=1e-12)
        for dtype, tol in [(np.float64, 1e-12), (np.float32, 2e-6)]:
            x, y = a.astype(dtype), b.astype(dtype)
            np.testing.assert_array_equal(v, np.abs(y[:, 2]) > 1e-8)
            np.testing.assert_array_equal(x[:, 3:], y[:, 3:])
            rays_x, rays_y = x[v, :2] / x[v, 2:3], y[v, :2] / y[v, 2:3]
            np.testing.assert_allclose(rays_x, rays_y, rtol=tol, atol=tol)
            if dtype == np.float32:
                maximum_ray_change = max(maximum_ray_change, float(np.max(np.abs(rays_x-rays_y))))
    assert len(m['samples']) == len(source) == 195
    result = dict(status='PASS', saved_clouds_verified=195,
                          manifest_and_protocol_hashes_verified=True,
                          float32_max_absolute_ray_change=maximum_ray_change,
                          manifest_sha256=sha(path), verifier_sha256=sha(Path(__file__)),
                          training_or_accuracy_evaluation_performed=False)
    (HOME / 'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
