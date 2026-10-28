"""Task-internal CC-inspired feature modulation, with a frozen depth backbone.

The module is an engineering prototype inspired by feature-wise modulation.
It does not claim calibrated physical illuminant estimation or novel FiLM.
"""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'external/Depth-Anything-V2/metric_depth'))

def frozen_model(threads=8):
    from depth_anything_v2.dpt import DepthAnythingV2
    torch.set_num_threads(threads)
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    model.load_state_dict(torch.load(ROOT/'checkpoints/depth_anything_v2_metric_hypersim_vits.pth',map_location='cpu',weights_only=True))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

class ColorFeatureAdapter(nn.Module):
    def __init__(self,channels=384,layers=4,hidden=32,strength=.25):
        super().__init__();self.channels=channels;self.layers=layers;self.strength=strength
        self.network=nn.Sequential(nn.Linear(3,hidden),nn.GELU(),nn.Linear(hidden,layers*channels*2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,features,condition):
        params=torch.tanh(self.network(condition)).reshape(-1,self.layers,2,self.channels)*self.strength
        adapted=[];penalty=params.square().mean()
        for i,(tokens,cls) in enumerate(features):
            gamma=params[:,i,0].unsqueeze(1);beta=params[:,i,1].unsqueeze(1)
            scale=tokens.detach().std(dim=(1,2),keepdim=True).clamp_min(.1)
            new=tokens*(1+gamma)+beta*scale
            adapted.append((new,cls))
        return adapted,penalty

def predict_from_features(model,features,patch_hw,adapter=None,condition=None,output_hw=None):
    penalty=torch.zeros((),dtype=features[0][0].dtype)
    if adapter is not None:features,penalty=adapter(features,condition)
    pred=model.depth_head(features,*patch_hw)[:,0]*model.max_depth
    if output_hw is not None:pred=F.interpolate(pred[:,None],size=output_hw,mode='bilinear',align_corners=True)[:,0]
    return pred,features,penalty

def invariant_feature_loss(features):
    terms=[]
    for tokens,_ in features:
        normalized=F.normalize(tokens,dim=-1)
        terms.append((normalized[0]-normalized[1]).square().sum(-1).mean())
    return torch.stack(terms).mean()
