"""Audit v2 uses the training loader contiguous tensor layout; no retraining."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import torch
from data import ROOT, MANIFESTS, DAMAGE, load, split, sha
from model import make_model

HOME=Path(__file__).resolve().parent
EVIDENCE=HOME/'pilot_v1'
RUNS=HOME/'runs/eth3d_pilot_v1'


def independent_angles(prediction,labels):
    a=np.asarray(prediction,dtype=np.float64); b=np.asarray(labels,dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    a=a/np.linalg.norm(a,axis=1,keepdims=True); b=b/np.linalg.norm(b,axis=1,keepdims=True)
    raw=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),(a*b).sum(1))
    floor=np.arccos(.999999)
    return np.rad2deg(np.clip(raw,floor,np.pi-floor))


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assert torch.xpu.is_available()
    cfg=json.loads((EVIDENCE/'frozen_run.json').read_text())
    assert all(sha(HOME/n)==h for n,h in cfg['source_sha256'].items())
    eval_path=EVIDENCE/'evaluation_manifest.json'
    assert sha(eval_path)==cfg['evaluation_manifest_sha256']
    evaluation=json.loads(eval_path.read_text())
    manifest,values=load('ETH3D')
    assert evaluation['source_manifest_sha256']==sha(ROOT/MANIFESTS['ETH3D'])
    assert evaluation['protocol_sha256']==sha(HOME/'PILOT_PROTOCOL.md')
    clouds={}
    for r in evaluation['samples']:
        for field in ('cloud','label'):assert sha(ROOT/r[field])==r[field+'_sha256']
        cloud=np.load(ROOT/r['cloud']).astype(np.float32)
        np.testing.assert_array_equal(cloud[:,3:],values[r['id']][0][:,3:])
        clouds[(r['condition'],r['id'])]=cloud
    assert len(clouds)==975
    rows=[]; jobs=[]; max_saved_angle_difference=0.
    for name,(mode,damaged,consistency) in cfg['conditions'].items():
        for fold in range(3):
            folder=RUNS/name/f'fold{fold}'
            complete=json.loads((folder/'complete.json').read_text())
            configuration=json.loads((folder/'config.json').read_text())
            assert complete['status']=='complete' and complete['epochs']==100
            assert complete['config_sha256']==sha(folder/'config.json')
            assert configuration['mode']==mode and configuration['fold']==fold
            assert configuration['seed']==20261116+fold and configuration['device']=='xpu'
            assert configuration['damage_training']==damaged and configuration['consistency']==consistency
            assert configuration['manifest_sha256']==sha(ROOT/MANIFESTS['ETH3D'])
            assert configuration['splits']==split(manifest,fold)
            assert all(sha(HOME/n)==h for n,h in configuration['sources_sha256'].items())
            history=json.loads((folder/'history.json').read_text())
            assert [r['epoch'] for r in history]==list(range(1,101))
            scores=np.array([r['validation_mean_deg'] for r in history]); assert np.isfinite(scores).all()
            assert np.isfinite([r['training_loss'] for r in history]).all()
            selected=int(np.argmin(scores))+1
            assert selected==complete['validation_selected_epoch']
            ids=configuration['splits']['test']
            for selection in ['fixed_final','validation_selected']:
                checkpoint=folder/f'{selection}.pt'
                assert sha(checkpoint)==complete['checkpoints_sha256'][checkpoint.name]
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert state['epoch']==(100 if selection=='fixed_final' else selected)
                assert state['config']==configuration
                net=make_model(mode).to('xpu'); net.load_state_dict(state['state_dict']); net.eval()
                saved=json.loads((folder/f'{selection}_predictions.json').read_text())
                assert saved['ids']==ids
                labels=np.stack([values[sid][1] for sid in ids])
                np.testing.assert_array_equal(labels,np.asarray(saved['label'],dtype=np.float32))
                saved_errors=independent_angles(saved['prediction'],labels)
                discrepancy=float(np.max(np.abs(saved_errors-saved['angular_error_deg'])))
                assert discrepancy<1e-8
                max_saved_angle_difference=max(max_saved_angle_difference,discrepancy)
                for condition in DAMAGE:
                    vectors=[]; gates=[]
                    with torch.inference_mode():
                        for start in range(0,len(ids),16):
                            selected_ids=ids[start:start+16]
                            x=torch.from_numpy(np.stack([clouds[(condition,sid)].T.copy() for sid in selected_ids])).to('xpu')
                            result=net(x)
                            vectors.extend(result['illuminant'].cpu().numpy().astype(float).tolist())
                            gates.extend(result['gate'].mean((1,2)).cpu().numpy().astype(float).tolist())
                    if condition=='clean':
                        np.testing.assert_array_equal(np.asarray(vectors),np.asarray(saved['prediction']))
                    errors=independent_angles(vectors,labels)
                    for sid,pred,label,error,gate in zip(ids,vectors,labels.tolist(),errors,gates):
                        rows.append(dict(method=name,selection=selection,fold=fold,id=sid,condition=condition,
                                         prediction=pred,label=label,error_deg=float(error),mean_gate=gate))
                del net
            jobs.append(dict(method=name,fold=fold,config_sha256=sha(folder/'config.json'),
                             checkpoints_sha256=complete['checkpoints_sha256']))
            print(f'Audited {name}/fold{fold}',flush=True)
    assert len(jobs)==21 and len(rows)==13650
    summary=[]
    for method in cfg['conditions']:
        for selection in ['fixed_final','validation_selected']:
            for condition in DAMAGE:
                subset=[r for r in rows if r['method']==method and r['selection']==selection and r['condition']==condition]
                assert len(subset)==len({r['id'] for r in subset})==195
                errors=np.sort([r['error_deg'] for r in subset])
                summary.append(dict(method=method,selection=selection,condition=condition,n=195,
                                    mean=float(errors.mean()),median=float(np.median(errors)),
                                    worst25=float(errors[-(len(errors)//4):].mean())))
    all_ids=[r['id'] for r in manifest['samples']]
    def capture_errors(method,conditions):
        return np.array([np.mean([r['error_deg'] for r in rows if r['method']==method and r['selection']=='fixed_final' and r['id']==sid and r['condition'] in conditions]) for sid in all_ids])
    rng=np.random.default_rng(20261118); pairs=[]
    for condition,which in [('clean',['clean']),('average_damaged',list(DAMAGE[1:]))]:
        delta=capture_errors('learned',which)-capture_errors('mask',which)
        boot=np.array([delta[rng.integers(0,195,195)].mean() for _ in range(10000)])
        pairs.append(dict(contrast='learned minus mask',endpoint=condition,role='primary',mean_deg=float(delta.mean()),
                          conditional_capture_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),n=195))
    per_fold=[]
    for method in cfg['conditions']:
        for selection in ['fixed_final','validation_selected']:
            for fold in range(3):
                for condition in DAMAGE:
                    errors=[r['error_deg'] for r in rows if r['method']==method and r['selection']==selection and r['fold']==fold and r['condition']==condition]
                    per_fold.append(dict(method=method,selection=selection,fold=fold,condition=condition,n=len(errors),mean=float(np.mean(errors))))
    for name,data in [('summary',summary),('per_fold',per_fold)]:
        dump(EVIDENCE/f'{name}.json',data)
        with (EVIDENCE/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    dump(EVIDENCE/'predictions.json',rows);dump(EVIDENCE/'paired.json',pairs)
    dump(EVIDENCE/'audit.json',dict(status='PASS',completed_models=21,checkpoints_reinferred=42,
          prediction_rows_independently_checked=len(rows),input_test_clouds=975,
          max_saved_angle_discrepancy_deg=max_saved_angle_difference,
          clean_saved_predictions_reproduced_exactly=True,all_histories_and_validation_selection_verified=True,
          frozen_run_sha256=sha(EVIDENCE/'frozen_run.json'),analyzer_sha256=sha(Path(__file__)), analyzer_file=Path(__file__).name,
          original_frozen_analyzer_sha256=sha(HOME/'analyze_pilot.py'),
          audit_layout_fix='Use .T.copy() exactly as data.batch before np.stack; model and frozen training/evaluation inputs unchanged',
          jobs=jobs,paid_compute_launched=False,limitation='100 epochs, one seed/fold, previously inspected captures, no scene independence claim'))
    lines=['# ETH3D reliability fusion pilot results','',
           '21 new models completed 100 epochs on the local Intel GPU. Three official folds; one seed per fold. Development pilot, not full paper-budget training.','',
           '| Method | Clean | Noise 10 | Noise 30 | Drop 50 | Shuffle |','|---|---:|---:|---:|---:|---:|']
    for method in cfg['conditions']:
        row=[next(r['mean'] for r in summary if r['method']==method and r['selection']=='fixed_final' and r['condition']==c) for c in DAMAGE]
        lines.append('| '+method+' | '+' | '.join(f'{v:.4f}' for v in row)+' |')
    lines+=['','All entries are mean recovery angular error in degrees; lower is better. Fixed-final checkpoint results above. Both checkpoint rules and all folds remain in the tables.','',
            '## Primary paired comparisons','']
    for pair in pairs:
        lo,hi=pair['conditional_capture_bootstrap95']
        lines.append(f"- {pair['endpoint']}: learned minus mask {pair['mean_deg']:+.4f} degrees, conditional 95% interval [{lo:+.4f}, {hi:+.4f}].")
    supported=all(p['conditional_capture_bootstrap95'][1]<0 for p in pairs)
    lines+=['',('Both primary intervals favor gating in this pilot; independent confirmation and full-budget training remain necessary.' if supported else 'This pilot does not establish an across-endpoint gating benefit. Report the two endpoints separately and retain all harms.'),'',
            'Intervals are conditional on these trained captures/models, not room or repeated-seed uncertainty; the two primary comparisons are not multiplicity adjusted. Synthetic depth damage is not measured sensor failure. Gate values are not calibrated reliability. No novelty or broad generalization is established.','',
            'Audit PASS: all 42 checkpoints re-inferred, 13,650 prediction rows checked, and all histories/validation selection/source/input/checkpoint identities verified. The earlier CPU/XPU gradient mismatch for original PCCC remains recorded; same-XPU repeatability passed, and all 21 models used XPU.','']
    (EVIDENCE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(audit='PASS',primary=pairs),indent=2))


if __name__=='__main__':main()
