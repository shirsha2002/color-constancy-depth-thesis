"""End-to-end worker test on synthetic fixtures, never benchmark accuracy."""
from pathlib import Path
import argparse
import json
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import train as worker
from data import split, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','xpu'],default='cpu')
    device=parser.parse_args().device
    rng = np.random.default_rng(87)
    ids = [f'fixture_{i}' for i in range(10)]
    values = {}
    for sid in ids:
        x = rng.uniform(.05, 1, size=(256, 6)).astype(np.float32)
        values[sid] = (x, np.array([.8, 1, .6], dtype=np.float32))
    manifest = dict(samples=[dict(id=s) for s in ids],
                    folds={'0':dict(train_ids=ids[:8], test_ids=ids[8:])})
    membership = split(manifest, 0)
    seen = set(); test_epochs = []
    real_batch = worker.batch
    with tempfile.TemporaryDirectory(prefix='fusion_worker_fixture_') as temporary:
        output = Path(temporary)/'run'
        def checked_batch(selected, *args, **kwargs):
            if torch.is_grad_enabled():
                assert set(selected) <= set(membership['train'])
                seen.update(selected)
            elif set(selected) & set(membership['test']):
                history = json.loads((output/'history.json').read_text())
                test_epochs.append(len(history))
                assert len(history) == 2
            return real_batch(selected, *args, **kwargs)
        argv = ['train.py','--dataset','ETH3D','--mode','learned','--epochs','2',
                '--damage-training','--execute-training','--device',device,'--out',str(output)]
        with patch.object(worker, 'load', return_value=(manifest, values)), \
             patch.object(worker, 'batch', side_effect=checked_batch), patch('sys.argv',argv):
            worker.main()
        complete = json.loads((output/'complete.json').read_text())
        assert complete['status']=='complete' and complete['epochs']==2
        for name, expected in complete['checkpoints_sha256'].items():
            assert sha(output/name)==expected
        for selection in ['fixed_final','validation_selected']:
            p = json.loads((output/f'{selection}_predictions.json').read_text())
            assert p['ids']==membership['test'] and np.isfinite(p['angular_error_deg']).all()
        assert seen==set(membership['train']) and test_epochs==[2,2]
        # Independent checkpoint inference must reproduce the saved vectors.
        checkpoint = torch.load(output/'fixed_final.pt',weights_only=True,map_location='cpu')
        net = worker.make_model('learned').to(device); net.load_state_dict(checkpoint['state_dict'])
        repeated = worker.score(net,membership['test'],values,device)
        saved = json.loads((output/'fixed_final_predictions.json').read_text())
        np.testing.assert_array_equal(repeated['prediction'],saved['prediction'])
    result = dict(status='PASS_SYNTHETIC_WORKER_TEST', synthetic_fixture_only=True,device=device,
                  optimizer_accessed_training_ids_only=True, outer_test_access_after_all_epochs=True,
                  completion_and_checkpoint_hashes_verified=True, saved_prediction_reinference_exact=True,
                  worker_sha256=sha(Path(worker.__file__)), test_sha256=sha(Path(__file__)),
                  benchmark_performance_claim=False, paid_compute_launched=False)
    path=Path(__file__).with_name('worker_checks.json' if device=='cpu' else 'worker_checks_xpu.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
