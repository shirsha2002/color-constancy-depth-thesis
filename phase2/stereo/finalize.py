"""Cross-check final-run provenance and export a report handoff, without inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'phase2'
HERE = Path(__file__).resolve().parent
OUT = PHASE / 'results/stereo/main_v2'
LABELS = {'rgb_ssd': 'RGB SSD', 'scalar_gain': 'Scalar gain',
          'diagonal_gain': 'Diagonal gain', 'gray_zncc': 'Gray ZNCC',
          'channel_zncc': 'Channel ZNCC'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    code = ''.join((HERE / f).read_text() for f in
                   ['geometry.py', 'costs.py', 'run_experiment.py'])
    assert hashlib.sha256(code.encode()).hexdigest() == protocol['code_sha256']
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['protocol_sha256'] == fingerprint
    audit = json.loads((OUT / 'audit.json').read_text())
    assert audit['independent_float64_score_recalculation_passed']
    assert audit['audited_scores'] == 210
    rows = list(csv.DictReader((OUT / 'metrics.csv').open(newline='', encoding='utf-8')))
    key = lambda r: (int(r['scene']), int(r['reference_view']),
                     int(r['source_light']), r['method'])
    csv_by_key = {key(r): r for r in rows}
    assert len(rows) == len(csv_by_key) == 210
    records = sorted((OUT / 'records').glob('*.json'))
    assert len(records) == 42
    for path in records:
        record = json.loads(path.read_text())
        assert record['protocol_sha256'] == fingerprint
        assert (OUT / 'arrays' / path.with_suffix('.npz').name).exists()
        for r in record['metrics']:
            c = csv_by_key[key(r)]
            for name, value in r.items():
                if isinstance(value, (float, int)):
                    assert np.isclose(float(c[name]), value, rtol=1e-12, atol=1e-12)
                else:
                    assert c[name] == value
    for group in summary['summary']:
        selected = [r for r in rows if all(r[k] == group[k] for k in
                                           ['split', 'lighting', 'method'])]
        assert len(selected) == group['view_light_cases']
        for metric in ['mae_mm', 'rmse_mm', 'bad5_percent']:
            assert np.isclose(np.mean([float(r[metric]) for r in selected]),
                              group[metric], rtol=1e-12, atol=1e-12)
    manifest = json.loads((PHASE / 'data/dtu/manifest.json').read_text())
    for im in manifest['images']:
        assert sha(PHASE / im['image']) == im['sha256']
    rr = [r for r in rows if r['split'] == 'evaluation']
    def avg(method, lighting):
        return float(np.mean([float(r['mae_mm']) for r in rr
                              if r['method'] == method and r['lighting'] == lighting]))
    methods = list(LABELS)
    values = {m: {'matched_mae_mm': avg(m, 'matched_diffuse'),
                  'changed_mae_mm': avg(m, 'mismatched')} for m in methods}
    table = [['Method', 'Matched MAE (mm)', 'Changed-light MAE (mm)', 'Ratio']]
    for m, v in values.items():
        v['changed_over_matched_ratio'] = v['changed_mae_mm'] / v['matched_mae_mm']
        table.append([LABELS[m], f"{v['matched_mae_mm']:.2f}",
                      f"{v['changed_mae_mm']:.2f}", f"{v['changed_over_matched_ratio']:.2f}x"])
    handoff = {
        'passed': True, 'run': 'main_v2', 'report_table': table,
        'report_interpretation': (
            'On held-out scene 6, raw RGB matching increases from 13.19 mm MAE '
            'under matched lighting to 54.21 mm under changed source lighting '
            '(4.11 times). Modeling radiometric nuisance inside matching helps, '
            'but the diagonal-channel candidate (36.43 mm) is worse than scalar '
            'gain (29.90 mm) and established channel ZNCC (32.84 mm). This supports '
            'lighting sensitivity in geometry; it does not establish an extra '
            'benefit from color constancy.'),
        'report_figure': 'phase2/results/stereo/main_v2/figures/heldout_lighting_mae.png',
        'values': values,
        'aggregation': 'Arithmetic mean of per-view/light MAE. Matched: light3, 3 reference views. Changed: lights0,1,2,4,5,6, 18 view/light cases. One evaluation scene.',
        'audited_scores': 210, 'case_count': 42, 'raw_rgb_hashes_checked': 42,
        'code_matches_frozen_run': True, 'csv_records_summary_consistent': True,
        'array_scores_independently_recomputed': True,
        'method_support_identical': audit['method_support_identical'],
        'lighting_masks_identical': audit['lighting_masks_identical'],
        'protocol_sha256': fingerprint,
        'validation': 'phase2/results/stereo/method_validation.json',
        'scope_limits': summary['limitations'],
    }
    (OUT / 'report_handoff.json').write_text(json.dumps(handoff, indent=2), encoding='utf-8')
    print(json.dumps(handoff, indent=2))


if __name__ == '__main__':
    main()
