"""Check reusable baseline identities and the actual input loader, without training."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    jobs = []
    frozen_sources = {}
    for condition in ['ETH3D_real', 'ETH3D_flat']:
        folder = ROOT / 'phase6_modal/results/bc_t4_v1' / condition
        cfg_path = folder / 'config.json'
        cfg = json.loads(cfg_path.read_text())
        assert cfg['epochs'] == 10000 and cfg['device'] == 'cuda'
        assert cfg['variants']['released_real'] == ['released', 'real', .2]
        manifest = json.loads((ROOT / cfg['manifest_path']).read_text())
        assert sha(ROOT / cfg['manifest_path']) == cfg['manifest_sha256']
        for name, expected in cfg['sources'].items():
            assert sha(ROOT / name) == expected
            frozen_sources[name] = expected
        for name, expected in cfg['upstream']['files'].items():
            assert sha(ROOT / 'external/Point-Cloud-Color-Constancy' / name) == expected
        for fold in range(3):
            job = folder / f'released_real_fold{fold}'
            complete = json.loads((job / 'complete.json').read_text())
            assert complete['status'] == 'complete' and complete['epochs'] == 10000
            assert complete['configuration_sha256'] == cfg['sha256']
            checkpoint = job / 'checkpoints/fixed_final.pt'
            expected = complete['checkpoint_sha256']['fixed_final.pt']
            assert sha(checkpoint) == expected
            pred = json.loads((job / 'fixed_final_predictions.json').read_text())
            assert pred['selected_epoch'] == 10000
            assert pred['configuration_sha256'] == cfg['sha256']
            assert pred['ids'] == manifest['folds'][str(fold)]['test_ids']
            jobs.append(dict(condition=condition, fold=fold,
                             config_file_sha256=sha(cfg_path),
                             checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                             checkpoint_sha256=expected))
    # Import only the existing tensor loader. Do not instantiate/train a model.
    sys.path.insert(0, str(ROOT / 'phase4_pccc'))
    import pilot
    path = HOME / 'inputs/ETH3D_unitdepth/manifest.json'
    m = json.loads(path.read_text())
    values = {r['id']: (np.load(ROOT / r['cloud']).astype(np.float32),
                       np.load(ROOT / r['label']).astype(np.float32)) for r in m['samples']}
    ids = [r['id'] for r in m['samples']]
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        x, y = pilot.tensor_batch(selected, values, geometry='real', training=False)
        assert tuple(x.shape) == (len(selected), 6, 256)
        np.testing.assert_array_equal(x.numpy(), np.stack([values[i][0].T for i in selected]))
        np.testing.assert_array_equal(y.numpy(), np.stack([values[i][1] for i in selected]))
    result = dict(status='PASS_LOCAL_READINESS', reusable_final_baselines=jobs,
                  frozen_numerical_sources=frozen_sources, loader_captures_checked=len(ids),
                  new_manifest_sha256=sha(path), checker_sha256=sha(Path(__file__)),
                  model_instantiated=False, training_launched=False,
                  cloud_launch_ready=False,
                  remaining=['Bounded execution runner, new timing/cost allowance and explicit new spending authorization',
                             'Original RGB/correspondence prerequisites for separate predicted-depth study'])
    (HOME / 'readiness.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reusable_final_baselines','frozen_numerical_sources']}, indent=2))


if __name__ == '__main__':
    main()
