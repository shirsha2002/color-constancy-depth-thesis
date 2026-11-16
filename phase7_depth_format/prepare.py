"""Freeze three geometry conditions without changing old data or numerical code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'phase7_depth_format'
CONDITIONS = ['camera_real', 'camera_unitdepth', 'camera_flat']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(path, value):
    payload = (json.dumps(value, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == payload, f'Frozen file differs: {path}'
    else:
        path.write_bytes(payload)


def transform(x, condition):
    assert x.shape == (256, 6) and np.isfinite(x).all()
    valid = np.abs(x[:, 2]) > 1e-8
    assert valid.sum() >= 16 and np.all(x[~valid, :3] == 0)
    index = np.arange(256)[valid]
    rays = x[valid, :2] / x[valid, 2:3]
    correlations = [float(np.corrcoef(rays[:, 0], index % 16)[0, 1]),
                    float(np.corrcoef(rays[:, 1], index // 16)[0, 1])]
    assert min(abs(c) for c in correlations) > .99
    zsign = int(np.sign(np.median(x[valid, 2])))
    assert np.all(np.sign(x[valid, 2]) == zsign)
    axes = np.array([int(np.sign(correlations[0])) * zsign,
                     int(np.sign(correlations[1])) * zsign, zsign])
    assert int(np.prod(axes)) == 1, 'Must be a proper camera-axis rotation'
    out = x.copy()
    out[valid, :3] *= axes
    scale = float(np.median(np.abs(x[valid, 2])))
    if condition == 'camera_unitdepth':
        out[valid, :3] /= scale
    elif condition == 'camera_flat':
        out[valid, :3] /= out[valid, 2:3].copy()
    else:
        assert condition == 'camera_real'
    expected_rays = rays * axes[:2] / axes[2]
    np.testing.assert_allclose(out[valid, :2] / out[valid, 2:3], expected_rays, rtol=1e-12, atol=1e-12)
    assert np.array_equal(out[:, 3:], x[:, 3:]) and np.array_equal(out[~valid, :3], x[~valid, :3])
    assert (out[valid, 2] > 0).all()
    if condition == 'camera_unitdepth':
        assert abs(np.median(out[valid, 2]) - 1) < 1e-12
        np.testing.assert_allclose(out[valid, 2] / np.median(out[valid, 2]), np.abs(x[valid, 2]) / scale)
    if condition == 'camera_flat':
        np.testing.assert_allclose(out[valid, 2], 1)
    return out, {'axis_rotation_diagonal': axes.tolist(), 'native_median_abs_z': scale,
                 'native_ray_correlations': correlations, 'valid_points': int(valid.sum())}


def main():
    source = ROOT / 'phase6_modal/inputs/NYU_DIODE_real/manifest.json'
    native = json.loads(source.read_text())
    groups = {}
    checks = 0
    for condition in CONDITIONS:
        records = []
        for r in native['samples']:
            for kind in ['cloud', 'label']:
                assert sha(ROOT / r[kind]) == r[kind + '_sha256']
            x = np.load(ROOT / r['cloud']).astype(np.float64)
            y, info = transform(x, condition)
            path = HOME / 'inputs' / condition / 'clouds' / f"{r['id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert np.array_equal(np.load(path), y)
            else:
                np.save(path, y)
            rec = dict(r, cloud=path.relative_to(ROOT).as_posix(), cloud_sha256=sha(path),
                       native_cloud=r['cloud'], native_cloud_sha256=r['cloud_sha256'], transform=info)
            records.append(rec)
            checks += 1
            if condition == CONDITIONS[0]:
                group = 'DIODE' if 'door' in r['id'] else 'NYU'
                groups.setdefault(group, []).append(info)
        manifest = {'condition': condition, 'dataset': 'NYU_DIODE', 'points_per_cloud': 256,
                    'source_manifest': source.relative_to(ROOT).as_posix(), 'source_manifest_sha256': sha(source),
                    'prepare_source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(HOME / 'PROTOCOL.md'),
                    'samples': records, 'folds': native['folds']}
        for f, fold in manifest['folds'].items():
            assert not set(fold['train_ids']) & set(fold['test_ids'])
            assert len(fold['train_ids']) + len(fold['test_ids']) == 324
        freeze(HOME / 'inputs' / condition / 'manifest.json', manifest)
    preflight = {'status': 'PASS', 'transformed_clouds_checked': checks, 'samples': 324,
                 'conditions': CONDITIONS, 'rgb_bitwise_unchanged': True, 'labels_unchanged': True,
                 'canonical_rays_and_relative_depth_checked': True, 'proper_rotations_only': True,
                 'groups': {k: {'n': len(v), 'rotations': sorted({tuple(i['axis_rotation_diagonal']) for i in v}),
                                'median_native_abs_z': float(np.median([i['native_median_abs_z'] for i in v]))}
                            for k, v in groups.items()},
                 'protocol_sha256': sha(HOME / 'PROTOCOL.md')}
    freeze(HOME / 'preflight.json', preflight)
    rate = .000164 + 4 * .0000131 + 4 * .00000222
    freeze(HOME / 'budget.json', {'approved_usd': 8, 'approval_source': 'User explicitly answered US $8 in chat',
           'checked_date': '2026-11-16', 'pricing_source': 'https://modal.com/pricing',
           't4_usd_per_second': .000164, 'cpu_usd_per_core_second': .0000131,
           'ram_usd_per_gib_second': .00000222, 'maximum_cpu_cores': 4, 'maximum_ram_gib': 4,
           'model_jobs': 9, 'timeout_seconds_per_job': 3000, 'smoke_timeout_seconds': 300,
           'compute_upper_planning_usd': round((9 * 3000 + 300) * rate, 5),
           'overhead_reserve_usd': 1.5, 'automatic_retries': 0,
           'invoice_note': 'Planning estimate; does not include unrelated account usage or constitute provider billing enforcement.'})
    print(json.dumps(preflight, indent=2))


if __name__ == '__main__':
    main()

