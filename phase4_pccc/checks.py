"""Preflight the actual author model, augmentation and flat geometry control."""
import json
import random
import numpy as np
import torch
from upstream import ROOT,load,make_model,predict,sha
from pilot import flatten,augment,angles,dump,HOME

def main():
    torch.set_num_threads(8);author,data,utils=load()
    manifest=json.loads((HOME/'manifest.json').read_text());records=manifest['samples']
    train=next(r for r in records if r['split']=='train')
    x=np.load(ROOT/train['cloud']).astype(np.float32);y=np.load(ROOT/train['label']).astype(np.float32)
    # The wrapper must produce exactly the original released training item.
    original=data.PcdColor.__new__(data.PcdColor)
    original.file_list=[train['id']+'.npy'];original.train=True;original.sta_aug=True;original.sizes=16
    original.data_path=str((ROOT/train['cloud']).parent.parent/'pcd')
    original.gt_path=str((ROOT/train['label']).parent)+'/'
    random.seed(17);np.random.seed(17);expected,expected_label,_=original[0]
    random.seed(17);np.random.seed(17);actual=augment(x,original,.2)
    assert np.array_equal(actual.T,expected.numpy()) and np.array_equal(y,expected_label.numpy())
    flat=flatten(x);valid=np.abs(x[:,2])>1e-8
    assert np.allclose(flat[valid,:2]/flat[valid,2:3],x[valid,:2]/x[valid,2:3],atol=1e-6)
    assert np.array_equal(flat[:,3:],x[:,3:]) and (flat[~valid,:3]==0).all()
    batch=torch.from_numpy(np.stack([x.T.copy(),x.T.copy()]))
    params={}
    for head in ['released_sum','paper_weighted']:
        torch.manual_seed(8);model=make_model(author,head);model.eval()
        local,_=model(batch)
        if head=='released_sum':expected=local.sum(2)
        else:
            weights=torch.softmax(local[:,3:4],2)
            assert torch.allclose(weights.sum(2),torch.ones(2,1))
            expected=(local[:,:3]*weights).sum(2)
        assert torch.equal(predict(model,batch,head),expected)
        loss=utils.get_angular_loss(expected,torch.from_numpy(np.stack([y,y])))
        assert torch.isfinite(loss);loss.backward()
        assert model.conv4.weight.grad is not None and torch.isfinite(model.conv4.weight.grad).all()
        params[head]=sum(p.numel() for p in model.parameters())
    assert abs(angles([[1,0,0]],[[0,1,0]])[0]-90)<1e-9
    assert abs(angles([[1,0,0]],[[1,0,0]])[0]-np.degrees(np.arccos(.999999)))<1e-9
    splits={s:{r['id'] for r in records if r['split']==s} for s in ['train','validation','test']}
    assert not splits['train']&splits['validation'] and not splits['train']&splits['test'] and not splits['validation']&splits['test']
    result={'preflight':'PASS','native_augmentation_bitwise_matches_author':True,
        'flat_geometry_preserves_valid_rays_rgb_and_missing_mask':True,'head_aggregation_checks':True,
        'finite_actual_angular_gradients':True,'parameter_counts':params,
        'manifest_sha256':sha(HOME/'manifest.json'),'split_counts':{k:len(v) for k,v in splits.items()}}
    dump(HOME/'preflight.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
