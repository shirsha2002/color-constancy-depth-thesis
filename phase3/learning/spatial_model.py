"""Local relative-chroma transport guided by predicted geometry.

An experimental coupling, not a claim that its ingredients are novel. No
reference depth or calibrated test geometry enters this RGB-only forward path.
"""
import torch
from torch import nn
import torch.nn.functional as F

MODES=('spatial_content','spatial_flat','spatial_geometry')

class SpatialGeometryAdapter(nn.Module):
    def __init__(self,channels=384,layers=4,rank=8,hidden=32):
        super().__init__();self.channels=channels;self.layers=layers
        self.task_gamma=nn.Parameter(torch.zeros(layers,channels))
        self.task_beta=nn.Parameter(torch.zeros(layers,channels))
        self.semantic=nn.Linear(channels,hidden)
        self.photo=nn.Linear(6,hidden)
        self.geometry=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,hidden))
        self.context=nn.Sequential(nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.illumination=nn.Linear(hidden,2)
        self.confidence=nn.Linear(hidden,1)
        self.condition=nn.Linear(2,rank,bias=False)
        self.down=nn.ModuleList(nn.Linear(channels,rank,bias=False) for _ in range(layers))
        self.up=nn.ModuleList(nn.Linear(rank,channels,bias=False) for _ in range(layers))
        self.smoothing=nn.Parameter(torch.zeros(()))
        self.register_buffer('luma',torch.tensor([.2126,.7152,.0722]))
        nn.init.normal_(self.illumination.weight,std=.02);nn.init.zeros_(self.illumination.bias)
        nn.init.zeros_(self.confidence.weight);nn.init.constant_(self.confidence.bias,-1.)
        for layer in self.up:nn.init.zeros_(layer.weight)

    @staticmethod
    def neighbors(values,hw):
        b,n,c=values.shape;h,w=hw;assert n==h*w
        image=values.transpose(1,2).reshape(b,c,h,w)
        padded=F.pad(image,(1,1,1,1),mode='replicate')
        return F.unfold(padded,kernel_size=3).reshape(b,c,9,n)

    def forward(self,features,rgb_linear,base_depth,patch_hw,mode):
        if mode not in MODES:raise ValueError(mode)
        h,w=patch_hw;first=features[0][0];batch,n,channels=first.shape
        assert n==h*w and channels==self.channels and len(features)==self.layers
        rgb=F.adaptive_avg_pool2d(rgb_linear.detach(),(h,w)).flatten(2).transpose(1,2).clamp_min(1e-5)
        predicted=F.interpolate(base_depth.detach()[:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0].clamp_min(.001)
        z=predicted/predicted.flatten(1).median(1).values[:,None,None].clamp_min(.001)
        if mode=='spatial_flat':z=torch.ones_like(z)
        z=z.clamp(.05,20.)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
        xx=xx.to(first);yy=yy.to(first)
        xyz=torch.stack((xx[None]*z,yy[None]*z,z),-1).reshape(batch,n,3)
        log_color=rgb.log();ratios=torch.stack((log_color[:,:,0]-log_color[:,:,1],log_color[:,:,2]-log_color[:,:,1]),-1)
        lum=(rgb*self.luma).sum(-1,keepdim=True).clamp_min(1e-5).log()
        photo=torch.cat((rgb,ratios,lum),-1)
        semantic=torch.stack([F.layer_norm(token,(channels,)) for token,_ in features]).mean(0)
        if mode=='spatial_content':photo=torch.zeros_like(photo)
        context=self.context(self.semantic(semantic)+self.photo(photo)+self.geometry(xyz))
        raw=self.illumination(context)+(ratios if mode!='spatial_content' else torch.zeros_like(ratios))
        logz=z.log().reshape(batch,n,1)
        affinity=torch.exp(-(self.neighbors(logz,(h,w))-logz.transpose(1,2)[:,:,None,:]).abs()/.15)
        adjacent=self.neighbors(raw,(h,w))
        mean=(adjacent*affinity).sum(2)/affinity.sum(2).clamp_min(1e-8)
        mean=mean.transpose(1,2)
        local=raw+torch.sigmoid(self.smoothing)*(mean-raw)
        graph_penalty=((adjacent-raw.transpose(1,2)[:,:,None,:]).abs()*affinity).sum()/(2*affinity.sum()).clamp_min(1e-8)
        modulation=torch.tanh(self.condition(.75*torch.tanh(local)))
        confidence=torch.sigmoid(self.confidence(context))
        gamma=.25*torch.tanh(self.task_gamma);beta=.25*torch.tanh(self.task_beta)
        output=[];regularizers=[gamma.square().mean(),beta.square().mean()]
        for k,(token,cls) in enumerate(features):
            scale=token.detach().std(dim=(1,2),unbiased=False,keepdim=True).clamp_min(.1)
            residual=torch.tanh(self.up[k](self.down[k](F.layer_norm(token,(channels,)))*modulation))
            adapted=token*(1+gamma[k][None,None])+beta[k][None,None]*scale+.1*confidence*residual*scale
            output.append((adapted,cls));regularizers.append(residual.square().mean())
        return output,{'illumination':local.mean(1),'local_illumination':local,
            'geometry_graph_penalty':graph_penalty,'regularization':torch.stack(regularizers).mean(),
            'geometry_source':'constant virtual rays' if mode=='spatial_flat' else 'detached frozen prediction; normalized virtual rays'}
