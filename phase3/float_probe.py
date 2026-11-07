"""Frozen-model causal probe with float photometry at the final input grid.

Added after a pixel-only precision audit, before depth test-score inspection.
No learner or checkpoint is changed. This supplements the locked quantized run.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase3/learning'))
import study

def main():
    torch.set_num_interop_threads(1);model=study.frozen_model(4)
    out=ROOT/'phase3/results/float_probe';out.mkdir(parents=True,exist_ok=True)
    records=[r for r in study.inputs()['nyu'] if r['split']=='evaluation']
    cfg={'protocol':'Float linear-luminance interventions after fixed224pixel resampling; before ImageNet normalization',
        'source_sha256':study.sha(Path(__file__)),'nyu_manifest_sha256':study.sha(ROOT/study.MANIFESTS['nyu']),
        'base_checkpoint_sha256':study.sha(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'),
        'frozen_conditions':study.EVAL_CONDITIONS,'training_or_checkpoint_selection':False,
        'note':'No uint8 rounding after interventions. Native resized RGB is clipped tovalid[0,1] before constructing the controlled field; all arms sharegeometry.'}
    path=out/'config.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:study.dump(path,cfg)
    rows=[];drifts=[];pre=[]
    mean=np.array([.485,.456,.406],np.float64);std=np.array([.229,.224,.225],np.float64)
    for index,r in enumerate(records,1):
        source=np.asarray(Image.open(ROOT/r['rgb']).convert('RGB'),dtype=np.float32)/255
        original,_=model.image2tensor(np.round(source[:,:,::-1]*255).astype(np.uint8),224)
        h,w=original.shape[-2:]
        source=np.clip(cv2.resize(source,(w,h),interpolation=cv2.INTER_CUBIC),0,1)
        reference=study.target(r);inputs={}
        for condition in study.EVAL_CONDITIONS:
            changed,meta=study.corrupt(source,condition,r['key'])
            tensor,_=model.image2tensor((changed[:,:,::-1]*255).astype(np.float32),224)
            assert tuple(tensor.shape[-2:])==(h,w)
            decoded=np.clip(tensor[0].permute(1,2,0).numpy().astype(np.float64)*std+mean,0,1)
            lum=float(np.mean(study.srgb_to_linear(decoded)@study.LUMA))
            inputs[condition]=(lum,meta)
            folder=out/'predictions';folder.mkdir(exist_ok=True)
            prediction=folder/f'{r["id"]}__{condition}.npy'
            if prediction.exists():p=np.load(prediction)
            else:
                with torch.no_grad():
                    depth=model(tensor)
                    p=F.interpolate(depth[:,None],size=reference.shape,mode='bilinear',align_corners=True)[0,0].numpy().astype(np.float32)
                np.save(prediction,p)
            rows.append({'family':'nyu','id':r['id'],'scene':r['scene'],'condition':condition,'method':'frozen_float',**study.score(p,reference,'nyu')})
        baseline=inputs['control'][0]
        for condition in ['chroma_rb','chroma_green']:
            drifts.append(abs(inputs[condition][0]/baseline-1))
            pre.append(abs(inputs[condition][1]['mean_linear_luminance']-inputs['control'][1]['mean_linear_luminance']))
        if index%16==0:study.csvwrite(out/'metrics.csv',rows);print(f'Floating photometric probe: {index}/128scenes',flush=True)
    study.csvwrite(out/'metrics.csv',rows)
    assert max(drifts)<5e-5 and max(pre)<1e-10
    by={c:{r['scene']:r['abs_rel'] for r in rows if r['condition']==c} for c in study.EVAL_CONDITIONS}
    scenes=sorted(by['control']);neutral=np.array([by['control'][s] for s in scenes])
    chroma=np.array([.5*(by['chroma_rb'][s]+by['chroma_green'][s]) for s in scenes])
    exposure=np.array([by['exposure'][s] for s in scenes])
    rng=np.random.default_rng(20261107);draw=rng.integers(0,128,size=(5000,128))
    comparisons={}
    for name,changed in [('chromatic',chroma),('exposure',exposure)]:
        delta=changed-neutral
        comparisons[name]={'changed_mean_absrel':float(changed.mean()),'neutral_mean_absrel':float(neutral.mean()),
            'paired_change':float(delta.mean()),'bootstrap_95_interval':np.quantile(delta[draw].mean(1),[.025,.975]).tolist()}
    study.dump(out/'summary.json',{'scenes':128,'predictions':len(rows),'comparisons':comparisons,
        'actual_encoder_input_max_relative_luminance_drift':max(drifts),'pre_quantization_max_error':max(pre),
        'new_trained_models':False,'metrics_csv_sha256':study.sha(out/'metrics.csv')})
    print('Float probe complete; results saved for joint analysis.',flush=True)

if __name__=='__main__':main()
