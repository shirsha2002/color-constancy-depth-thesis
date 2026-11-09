"""Audit complete paper-budget jobs and aggregate official out-of-fold results."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import train
from model import load,make,forward

ROOT=train.ROOT;HOME=train.HOME
def angle(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.isfinite(a).all() and np.linalg.norm(a)>0 and np.linalg.norm(b)>0
    value=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))
    minimum=np.degrees(np.arccos(.999999));return float(np.clip(value,minimum,180-minimum))
def metrics(errors):
    x=np.sort(np.asarray(errors,dtype=np.float64));n=len(x)
    return {'n':n,'mean_deg':float(x.mean()),'median_deg':float(np.median(x)),
        'trimean_deg':float((np.percentile(x,25)+2*np.median(x)+np.percentile(x,75))/4),
        'best25_deg':float(x[:int(.25*n)].mean()),'worst25_deg':float(x[int(.75*n):].mean()),
        'percentile95_deg':float(np.percentile(x,95))}
def csvfile(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variants',default='weighted_real,weighted_flat,released_real,released_flat');p.add_argument('--folds',default='0,1,2');args=p.parse_args()
    out=args.out.resolve();cfg=json.loads((out/'config.json').read_text())
    variants=args.variants.split(',');folds=[int(v) for v in args.folds.split(',')]
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for name,expected in cfg['sources'].items():assert train.sha(ROOT/name)==expected
    assert train.upstream.identities()==cfg['upstream']
    manifest=json.loads((ROOT/cfg['manifest_path']).read_text());assert train.sha(ROOT/cfg['manifest_path'])==cfg['manifest_sha256']
    by={r['id']:r for r in manifest['samples']};values={}
    for r in by.values():
        for name in ['cloud','label']:assert train.sha(ROOT/r[name])==r[name+'_sha256']
        values[r['id']]=(np.load(ROOT/r['cloud']).astype(np.float32),np.load(ROOT/r['label']).astype(np.float32))
    author,_,_=load();rows=[];checks=[];max_error=0.;max_prediction=0.
    for fold in folds:
        ids=manifest['folds'][str(fold)]['test_ids'];train_ids=manifest['folds'][str(fold)]['train_ids']
        for variant in variants:
            job=out/f'{variant}_fold{fold}';done=json.loads((job/'complete.json').read_text())
            assert done['configuration_sha256']==cfg['sha256'] and done['epochs']==cfg['epochs']
            history=json.loads((job/'history.json').read_text())
            assert len(history)==cfg['epochs'] and [r['epoch'] for r in history]==list(range(1,cfg['epochs']+1))
            best=min(history,key=lambda r:r['author_reference_heldout_loss_deg'])['epoch']
            assert best==done['author_reference_selected_epoch']
            head,geometry,_=cfg['variants'][variant]
            for selection in ['fixed_final','author_test_selected']:
                path=job/'checkpoints'/f'{selection}.pt';assert train.sha(path)==done['checkpoint_sha256'][path.name]
                state=torch.load(path,map_location='cpu',weights_only=True)
                assert state['selected_epoch']==(cfg['epochs'] if selection=='fixed_final' else best)
                assert state['configuration_sha256']==cfg['sha256']
                net=make(author,head,cfg['device']);net.load_state_dict(state['state_dict']);net.eval()
                saved=json.loads((job/f'{selection}_predictions.json').read_text());assert saved['ids']==ids
                with torch.inference_mode():
                    for start in range(0,len(ids),16):
                        x,y=train.batch(ids[start:start+16],values,geometry,cfg['device'])
                        actual=forward(net,x,head).cpu().numpy();recorded=np.array(saved['prediction'][start:start+16])
                        delta=float(np.max(np.abs(actual-recorded)));max_prediction=max(max_prediction,delta)
                        assert np.allclose(actual,recorded,rtol=1e-4,atol=2e-5),('Savedmodelpredictiondiff',variant,fold,selection,delta)
                for key,pred,label,error in zip(ids,saved['prediction'],saved['label'],saved['angular_error_deg']):
                    assert np.array_equal(np.array(label),values[key][1].astype(np.float64))
                    computed=angle(pred,label);max_error=max(max_error,abs(computed-error));assert abs(computed-error)<1e-7
                    rows.append({'id':key,'official_fold':fold,'method':variant,'selection':selection,
                        'selected_epoch':state['selected_epoch'],'pred_r':pred[0],'pred_g':pred[1],'pred_b':pred[2],
                        'gt_r':label[0],'gt_g':label[1],'gt_b':label[2],'angular_error_deg':computed})
                checks.append({'variant':variant,'fold':fold,'selection':selection,'selected_epoch':state['selected_epoch'],
                    'checkpoint_sha256':train.sha(path),'training_epochs':len(history)})
        labels=np.stack([values[key][1]/np.linalg.norm(values[key][1]) for key in train_ids]);mean=labels.mean(0)
        for method in ['gray_world_points','training_mean_illuminant']:
            for key in ids:
                cloud,label=values[key];pred=cloud[:,3:].mean(0) if method=='gray_world_points' else mean
                rows.append({'id':key,'official_fold':fold,'method':method,'selection':'none','selected_epoch':0,
                    'pred_r':float(pred[0]),'pred_g':float(pred[1]),'pred_b':float(pred[2]),
                    'gt_r':float(label[0]),'gt_g':float(label[1]),'gt_b':float(label[2]),'angular_error_deg':angle(pred,label)})
    expected=sum(len(manifest['folds'][str(f)]['test_ids']) for f in folds)
    assert len(rows)==expected*(len(variants)*2+2)
    grouped=defaultdict(list);perfold=defaultdict(list)
    for r in rows:grouped[(r['method'],r['selection'])].append(r);perfold[(r['method'],r['selection'],r['official_fold'])].append(r['angular_error_deg'])
    table=[]
    for (method,selection),items in grouped.items():
        assert len(items)==expected and len({r['id'] for r in items})==expected
        table.append({'method':method,'selection':selection,**metrics([r['angular_error_deg'] for r in items]),
            'macro_fold_mean_deg':float(np.mean([np.mean(e) for (m,s,f),e in perfold.items() if (m,s)==(method,selection)]))})
    csvfile(out/'predictions.csv',rows);csvfile(out/'summary.csv',table)
    csvfile(out/'per_fold_metrics.csv',[{'method':m,'selection':s,'fold':f,**metrics(e)} for (m,s,f),e in perfold.items()])
    audit={'audit':'PASS','trained_jobs':len(variants)*len(folds),'saved_checkpoints_reinferred':len(checks),
        'heldout_captures':expected,'prediction_rows':len(rows),'all_official_folds_complete':folds==[0,1,2],
        'max_independent_angle_difference_deg':max_error,'max_reinference_component_difference':max_prediction,
        'fixed_final_primary_is_not_test_selected':True,'author_reference_is_explicitly_test_selected':True,
        'configuration_sha256':cfg['sha256'],'predictions_csv_sha256':train.sha(out/'predictions.csv'),
        'auditor_source_sha256':train.sha(Path(__file__)),'checkpoint_checks':checks}
    train.dump(out/'independent_audit.json',audit)
    train.dump(out/'summary.json',{'scope':cfg['scope'],'device':cfg['device'],'epochs':cfg['epochs'],'folds':folds,
        'variants':variants,'table':table,'audit':'PASS','published_accuracy_reproduced_claimed':False})
    lines=['# DepthAWB paper-budget results','',f"Device: **{cfg['device']}**. Epochs per job: **{cfg['epochs']}**. Official folds: **{folds}**.",'',
        f'Completed {len(variants)*len(folds)} trained jobs and verified {len(rows)} prediction rows for {expected} held-out captures.','',
        '**Primary:** predeclared fixed-final epoch. **Author reference:** checkpoint selected on the held-out fold, matching the released entry point; this is optimistic and not an unbiased test estimate.','',
        '| Method | Checkpoint rule | Mean deg | Median | Trimean | Worst 25% |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['method']} | {r['selection']} | {r['mean_deg']:.4f} | {r['median_deg']:.4f} | {r['trimean_deg']:.4f} | {r['worst25_deg']:.4f} |")
    lines+=['','Lower angular error is better. Means pool out-of-fold captures; the CSV also reports equal-fold means.','',
        'The original CVPR paper reports DepthAWB256point mean1.05degrees. Exact equivalence remains limited by the released3channel/paper4channel discrepancy, restoration choices, preprocessing/calibration provenance and the geometric flat-control interpretation. Do not claim the entire paper or a novel algorithm has been reproduced from this table.','',
        'This study covers DepthAWB at256points. It does not include NYU/DIODE, ETH3D,4096points, all competing baselines or local multi-illuminant ground truth. Earlier pilot exposure means these are reproduction/development data, not new independent confirmation.','',
        'All numeric source/data hashes, checkpoint epochs and saved-model predictions were checked. Model weights remain local; identities, predictions and complete histories are retained. See `independent_audit.json`, `summary.csv`, `per_fold_metrics.csv` and `predictions.csv`.','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='checkpoint_checks'},indent=2),flush=True)

if __name__=='__main__':main()
