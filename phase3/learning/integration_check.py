"""Check the real frozen encoder/decoder gradient path using training data only."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import study
from model import GeometryChromaticAdapter

def main():
    torch.set_num_interop_threads(1)
    args=argparse.Namespace(input_size=224,threads=8,steps=600)
    cfg=study.config(args);data=study.inputs();model=study.frozen_model(8)
    store=study.Store(model,cfg);record=next(r for r in data['nyu'] if r['split']=='training')
    items=[store.get(record,'original'),store.get(record,'train_chroma')]
    torch.manual_seed(study.SEED);adapter=GeometryChromaticAdapter()
    pred,aux=study.infer(model,adapter,items,'geometry_chromatic')
    with torch.no_grad():baseline,_=study.infer(model,None,items,None)
    assert torch.equal(pred,baseline), 'Identity initialization changed actual decoder output'
    target=torch.from_numpy(study.target(record))[None,None]
    target=F.interpolate(target,size=pred.shape[-2:],mode='nearest')[:,0].expand_as(pred)
    loss=(torch.abs(pred-target)/target.clamp_min(.1)).mean()+.2*F.smooth_l1_loss(
        aux['illumination'][1]-aux['illumination'][0],torch.tensor(items[1]['intervention']['log_gain']))
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert adapter.task_gamma.grad is not None and adapter.task_gamma.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.up.parameters())
    result={'integration_check':'PASS','input_record_role':'training only',
        'initial_pred_identical_to_frozen':True,'frozen_backbone_no_gradients':True,
        'adapter_gradient_path':True,'input_prediction_shape':list(pred.shape),
        'test_data_used':False,'configuration_sha256':cfg['configuration_sha256']}
    study.dump(study.ROOT/'phase3/results/integration_check.json',result)
    print(result)

if __name__=='__main__':main()
