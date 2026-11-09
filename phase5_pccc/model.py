"""Pinned PCCC author network with explicit CPU/XPU execution fixes."""
from pathlib import Path
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase4_pccc'))
import upstream

def load():
    author,data,utils=upstream.load()
    path=upstream.SOURCE/'pointnet/model.py';source=path.read_text(encoding='utf-8')
    unpack='x, trans = self.feat(x)';assert source.count(unpack)==1
    identity='        if x.is_cuda:\n            iden = iden.cuda()'
    assert source.count(identity)==4
    source=source.replace(unpack,'x, trans, _ = self.feat(x)')
    source=source.replace(identity,'        iden = iden.to(device=x.device, dtype=x.dtype)')
    module=types.ModuleType('pccc_author_portable');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module,data,utils

def make(author,head,device):
    result=author.PCCC(k=3,sis=256,feature_transform=False)
    # With exactly256inputpoints, AdaptiveAvgPool1d(256) is an identity.
    # XPU's generic backward lacks a deterministic implementation; bypass only
    # this mathematically redundant operation, with CPU output/gradient checks.
    result.adp=torch.nn.Identity()
    if head=='weighted':result.conv4=torch.nn.Conv1d(128,4,1)
    else:assert head=='released'
    return result.to(device)

def forward(net,x,head):
    assert x.shape[-1]==256,'Identity pooling fix is valid only for256pointinput'
    points,_=net(x)
    if head=='released':return points.sum(2)
    return (points[:,:3]*torch.softmax(points[:,3:4],dim=2)).sum(2)

def synchronize(device):
    if device=='xpu':torch.xpu.synchronize()
