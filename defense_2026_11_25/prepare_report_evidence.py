"""Collect frozen pilot evidence and reproduce held-out outputs on the local GPU."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
PILOT = ROOT / 'phase9_reliability_fusion'
sys.path.insert(0, str(PILOT))
from data import load, sha
from model import make_model


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def angle(pred, label):
    a, b = np.asarray(pred, dtype=float), np.asarray(label, dtype=float)
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -.999999, .999999))))


def main():
    evidence = PILOT / 'pilot_v1'
    verification = read(evidence / 'independent_verification.json')
    assert verification['status'] == 'PASS'
    assert all(sha(evidence / n) == h for n, h in verification['audited_artifact_sha256'].items())
    frozen = read(evidence / 'frozen_run.json')
    assert all(sha(PILOT / n) == h for n, h in frozen['source_sha256'].items())
    runs = PILOT / 'runs/eth3d_pilot_v1'
    histories, completions = {}, []
    for method in frozen['conditions']:
        histories[method] = []
        for fold in range(3):
            folder = runs / method / f'fold{fold}'
            complete, config, history = [read(folder / n) for n in ('complete.json', 'config.json', 'history.json')]
            assert complete['status'] == 'complete' and complete['epochs'] == 100
            assert sha(folder / 'config.json') == complete['config_sha256']
            assert [h['epoch'] for h in history] == list(range(1, 101))
            assert tuple(len(config['splits'][k]) for k in ['train', 'validation', 'test']) == (104, 26, 65)
            assert all(sha(folder / n) == h for n, h in complete['checkpoints_sha256'].items())
            histories[method].append(history)
            completions.append({'method': method, 'fold': fold, 'epochs': 100,
                                'history_sha256': sha(folder / 'history.json'),
                                'checkpoint_sha256': complete['checkpoints_sha256']})
    assert len(completions) == 21
    summary, paired, rows = [read(evidence / n) for n in ('summary.json', 'paired.json', 'predictions.json')]
    lookup = {(r['method'], r['condition'], r['id']): r for r in rows if r['selection'] == 'fixed_final'}
    manifest, values = load('ETH3D')
    ids = read(runs / 'learned/fold0/config.json')['splits']['test']
    example_ids = [min(ids), max(ids)]
    records = {(r['condition'], r['id']): r for r in read(evidence / 'evaluation_manifest.json')['samples']}
    conditions = ['clean', 'noise10', 'noise30', 'drop50', 'shuffle']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert torch.xpu.is_available(), 'Local Intel GPU required for the saved-model demonstration'
    maximum, checked, point_gates = 0., 0, {}
    for method in ['learned', 'mask', 'pccc']:
        folder = runs / method / 'fold0'
        state = torch.load(folder / 'fixed_final.pt', map_location='cpu', weights_only=True)
        assert state['epoch'] == 100
        net = make_model(method).to('xpu')
        net.load_state_dict(state['state_dict'])
        net.eval()
        for condition in conditions:
            for start in range(0, len(ids), 16):
                part, clouds = ids[start:start + 16], []
                for sid in part:
                    record = records[(condition, sid)]
                    assert sha(ROOT / record['cloud']) == record['cloud_sha256']
                    clouds.append(np.load(ROOT / record['cloud']).astype(np.float32).T.copy())
                with torch.inference_mode():
                    result = net(torch.from_numpy(np.stack(clouds)).to('xpu'))
                pred = result['illuminant'].cpu().numpy()
                gates = result['gate'].cpu().numpy()
                for i, sid in enumerate(part):
                    saved = lookup[(method, condition, sid)]
                    np.testing.assert_array_equal(pred[i].astype(float), saved['prediction'])
                    assert abs(angle(pred[i], saved['label']) - saved['error_deg']) < 1e-8
                    maximum = max(maximum, float(np.max(np.abs(pred[i] - saved['prediction']))))
                    checked += 1
                    if method == 'learned' and sid in example_ids:
                        point_gates[f'{sid}/{condition}'] = gates[i, 0].astype(float).tolist()
        del net
        print(f'PASS fresh local-GPU inference {method}/fold0 across all five conditions', flush=True)
    assert checked == 975 and maximum == 0.
    examples = []
    for sid in example_ids:
        clouds = {}
        for condition in conditions:
            record = records[(condition, sid)]
            cloud = np.load(ROOT / record['cloud']).astype(np.float32)
            clouds[condition] = {'xyzrgb': cloud.astype(float).tolist(), 'path': record['cloud'],
                                 'sha256': record['cloud_sha256'],
                                 'valid_depth_points': int((np.abs(cloud[:, 2]) > 1e-8).sum())}
        examples.append({'id': sid, 'fold': 0, 'clouds': clouds,
                         'predictions': {m: {c: lookup[(m, c, sid)] for c in conditions}
                                         for m in ['learned', 'mask', 'pccc']}})
    spec = importlib.util.spec_from_file_location('prior_report_data', ROOT / 'phase6_modal/report_data.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    historical = [{'dataset': prior.NAME[d], 'captures': prior.pccc(d, 'real')['n'],
                   'real': prior.pccc(d, 'real')['mean'], 'flat': prior.pccc(d, 'flat')['mean']}
                  for d in ['ETH3D', 'DepthAWB', 'NYU_DIODE']]
    prior_completions = [json.loads((p / 'complete.json').read_text()) for p in prior.job_dirs()]
    assert len(prior_completions) == 48 and {p['epochs'] for p in prior_completions} == {10000}
    result = {'prepared_date': '24 November 2026', 'histories': histories, 'completions': completions,
              'summary': summary, 'paired': paired, 'historical': historical,
              'historical_models': 48, 'historical_epochs': 10000,
              'example_selection': 'Lexicographically first and last official fold 0 test IDs, independent of prediction errors',
              'examples': examples, 'point_gates': point_gates,
              'fresh_inference': {'status': 'PASS', 'device': torch.xpu.get_device_name(0), 'backend': 'xpu',
                                  'torch': str(torch.__version__), 'predictions_reproduced_exactly': checked,
                                  'maximum_component_discrepancy': maximum, 'retrained_models': 0, 'modal_used': False},
              'audit': {k: v for k, v in read(evidence / 'audit.json').items() if k != 'jobs'},
              'independent_verification': verification,
              'evidence_sha256': {n: sha(evidence / n) for n in ['summary.json', 'paired.json', 'audit.json',
                                   'independent_verification.json', 'predictions.json', 'frozen_run.json']},
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = HOME / 'experiment_report_evidence.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'PASS', 'models_complete': len(completions), 'fresh_inference': result['fresh_inference'],
                      'example_ids': example_ids, 'output': str(target)}, indent=2))


if __name__ == '__main__':
    main()
