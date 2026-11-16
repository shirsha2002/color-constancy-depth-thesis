"""Audit all declared final models and write a complete, non-selected comparison."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/'phase7_depth_format'
OUT=HOME/'results/nyu_frame_v1'
CONDITIONS=['camera_real','camera_unitdepth','camera_flat']
BASE=ROOT/'phase6_modal/results/bc_t4_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def angles(pred,labels):
    a=np.asarray(pred,dtype=np.float64);b=np.asarray(labels,dtype=np.float64)
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    raw=np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1))
    floor=np.arccos(.999999)
    return np.rad2deg(np.clip(raw,floor,np.pi-floor))


def stats(values):
    x=np.sort(np.asarray(values,dtype=np.float64));q=len(x)//4
    a,b,c=np.percentile(x,[25,50,75])
    return dict(n=len(x),mean=float(x.mean()),median=float(b),trimean=float((a+2*b+c)/4),
                best25=float(x[:q].mean()),worst25=float(x[-q:].mean()))


def main():
    sys.path[:0]=[str(ROOT/'phase5_pccc'),str(ROOT/'phase4_pccc')]
    from model import load,make,forward
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    author,_,_=load()
    old_manifest=json.loads((ROOT/'phase6_modal/inputs/NYU_DIODE_real/manifest.json').read_text())
    manifests={c:json.loads((HOME/f'inputs/{c}/manifest.json').read_text()) for c in CONDITIONS}
    manifests.update(native=old_manifest,native_flat=json.loads((ROOT/'phase6_modal/inputs/NYU_DIODE_flat/manifest.json').read_text()))
    all_ids=[r['id'] for r in old_manifest['samples']]
    assert len(all_ids)==len(set(all_ids))==324
    errors={};rows=[];predictions=[];audits=[];max_error=0.;max_prediction=0.
    for condition,manifest in manifests.items():
        records={r['id']:r for r in manifest['samples']}
        root=(BASE/('NYU_DIODE_real' if condition=='native' else 'NYU_DIODE_flat')) if condition.startswith('native') else OUT/condition
        cfg=json.loads((root/'config.json').read_text())
        if condition in CONDITIONS:
            assert cfg['epochs']==10000 and cfg['condition']==condition
            assert sha(HOME/f'inputs/{condition}/manifest.json')==cfg['manifest_sha256']
            for name,expected in cfg['additional_sources'].items():assert sha(HOME/name)==expected
        for name,expected in cfg['sources'].items():assert sha(ROOT/name)==expected
        for r in records.values():
            for name in ['cloud','label']:assert sha(ROOT/r[name])==r[name+'_sha256']
        selected={selection:{} for selection in ['fixed_final','author_test_selected']}
        for fold in range(3):
            job=root/f'released_real_fold{fold}'
            complete=json.loads((job/'complete.json').read_text())
            assert complete['status']=='complete' and complete['epochs']==10000
            assert complete['configuration_sha256']==cfg['sha256']
            if condition in CONDITIONS:
                history=json.loads((job/'history.json').read_text())
                assert [h['epoch'] for h in history]==list(range(1,10001))
                reference=np.asarray([h['author_reference_heldout_loss_deg'] for h in history])
                assert np.isfinite(reference).all()
                assert np.isfinite([h['training_loss_deg'] for h in history]).all()
                assert int(np.argmin(reference))+1==complete['author_reference_selected_epoch']
                dump(job/'history_every100.json',dict(full_history_sha256=sha(job/'history.json'),
                     full_history_epochs=10000,rows=[h for h in history if h['epoch']==1 or h['epoch']%100==0]))
            for selection in selected:
                p=json.loads((job/f'{selection}_predictions.json').read_text())
                ids=p['ids'];assert ids==manifest['folds'][str(fold)]['test_ids']
                assert len(ids)==len(set(ids))
                assert p['configuration_sha256']==cfg['sha256']
                assert p['selected_epoch']==(10000 if selection=='fixed_final' else complete['author_reference_selected_epoch'])
                labels=np.stack([np.load(ROOT/records[i]['label']).astype(np.float32) for i in ids])
                assert np.array_equal(labels,np.asarray(p['label'],dtype=np.float32))
                independent=angles(p['prediction'],labels)
                diff=float(np.max(np.abs(independent-np.asarray(p['angular_error_deg']))))
                max_error=max(max_error,diff);assert diff<1e-8
                selected[selection].update(zip(ids,independent.tolist()))
                for i,vector,label,error in zip(ids,p['prediction'],p['label'],independent):
                    predictions.append(dict(condition=condition,fold=fold,selection=selection,id=i,
                                            prediction=vector,label=label,error_deg=float(error)))
                checkpoint=job/f'checkpoints/{selection}.pt'
                assert sha(checkpoint)==complete['checkpoint_sha256'][f'{selection}.pt']
                ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert ck['configuration_sha256']==cfg['sha256']
                if selection=='fixed_final':
                    assert ck['selected_epoch']==10000
                    net=make(author,'released','cpu');net.load_state_dict(ck['state_dict']);net.eval()
                    values=[]
                    with torch.inference_mode():
                        for k in range(0,len(ids),16):
                            x=torch.from_numpy(np.stack([np.load(ROOT/records[i]['cloud']).astype(np.float32).T for i in ids[k:k+16]]))
                            values.extend(forward(net,x,'released').numpy())
                    measured=np.stack(values)
                    original=np.asarray(p['prediction'])
                    component=float(np.max(np.abs(measured-original)))
                    angle_disagreement=float(np.max(np.abs(angles(measured,labels)-independent)))
                    max_prediction=max(max_prediction,component)
                    # CUDA2.5 versus CPU2.14 can differ in convolution accumulation.
                    # Report discrepancies; do not require bitwise cross-device agreement.
                    assert np.allclose(measured,original,rtol=1e-3,atol=2e-3)
                    assert angle_disagreement<.02
                    audits.append(dict(condition=condition,fold=fold,checkpoint_sha256=sha(checkpoint),
                                       max_component_difference_cpu_vs_cuda=component,
                                       max_angular_difference_cpu_vs_cuda_deg=angle_disagreement))
        for selection,e in selected.items():
            assert set(e)==set(all_ids)
            errors[f'{condition}:{selection}']=e
            for group in ['all','NYU','DIODE']:
                ids=all_ids if group=='all' else [i for i in all_ids if ('door' in i)==(group=='DIODE')]
                rows.append(dict(condition=condition,selection=selection,group=group,**stats([e[i] for i in ids])))
    rng=np.random.default_rng(20261116)
    contrasts=[('camera_unitdepth','native','primary'),('camera_unitdepth','camera_flat','primary'),
               ('camera_real','native','secondary'),('camera_unitdepth','camera_real','secondary'),
               ('camera_flat','native_flat','secondary')]
    pairs=[]
    for left,right,role in contrasts:
        a=errors[f'{left}:fixed_final'];b=errors[f'{right}:fixed_final']
        d=np.array([a[i]-b[i] for i in all_ids])
        boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])
        pairs.append(dict(contrast=f'{left} minus {right}',role=role,mean_deg=float(d.mean()),
                          conditional_capture_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
                          left_better_captures=int((d<0).sum()),n=len(d)))
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(OUT/'summary.json',rows);dump(OUT/'paired.json',pairs);dump(OUT/'predictions.json',predictions)
    budget=json.loads((HOME/'budget.json').read_text())
    jobs=[json.loads(p.read_text()) for p in OUT.glob('*/*/complete.json')]
    seconds=sum(j['training_seconds'] for j in jobs)
    audit=dict(status='PASS',new_trained_models=len(jobs),all_final_checkpoints_reinferred=len(audits),
               new_training_histories_and_reference_selection_verified=True,
               saved_predictions_checked=len(predictions),max_independent_atan2_discrepancy_deg=max_error,
               cpu_vs_cuda_reinference=audits,protocol_sha256=sha(HOME/'PROTOCOL.md'),
               analyzer_sha256=sha(Path(__file__)),training_gpu_hours=seconds/3600,
               training_resource_planning_usd_upper=seconds*(budget['t4_usd_per_second']+4*budget['cpu_usd_per_core_second']+4*budget['ram_usd_per_gib_second']),
               billing_note='Estimated compute allowance, not the provider invoice; smoke/build overhead separate.')
    assert len(jobs)==9 and len(audits)==15
    dump(OUT/'audit.json',audit)
    final=[r for r in rows if r['selection']=='fixed_final' and r['group']=='all']
    text=['# NYU + DIODE camera-frame and scale experiment','',
          'Shirsha Mitra / 2023521460130 / 16 November 2026.','',
          'Nine new released-head PCCC models, 10000 epochs, official three folds. '
          'Old native and native-flat models reused; no gate or new architecture implemented.','',
          '| Geometry | Mean angular error | Median | Worst25% |','|---|---:|---:|---:|']
    text += [f"| {r['condition']} | {r['mean']:.4f} deg | {r['median']:.4f} | {r['worst25']:.4f} |" for r in final]
    text += ['','## Predeclared paired contrasts','',
             '| Contrast | Role | Mean difference | Conditional95% interval |','|---|---|---:|---|']
    text += [f"| {p['contrast']} | {p['role']} | {p['mean_deg']:+.4f} | [{p['conditional_capture_bootstrap95'][0]:+.4f}, {p['conditional_capture_bootstrap95'][1]:+.4f}] |" for p in pairs]
    text += ['','Negative differences favor the first condition. All contrasts are reported; '
             'intervals are conditional on these trained folds/captures, not independent rooms or repeated seeds. '
             'The two primary comparisons are not multiplicity adjusted. These are development data.','',
             'Per-source NYU/DIODE summaries and separately labelled test-selected references are in summary.csv. '
             'A coordinate normalization result is not physical calibration, proof of a new fusion algorithm, '
             'or a guarantee that geometry always helps.','',
             f'Independent audit: PASS. New training time {seconds/3600:.3f} GPU-hours. '
             'See audit.json for checkpoint verification and cross-backend inference differences.','']
    (HOME/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    print(json.dumps(dict(final=final,paired=pairs,audit_status='PASS'),indent=2))


if __name__=='__main__':main()

