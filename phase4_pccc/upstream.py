"""Load the pinned author release; keep its working tree unmodified.

Only one execution fix is applied in memory: PointNetfeat returns three values,
whereas the released PCCC.forward expects two. The paper-weighted head is an
explicit restoration variant, not represented as unchanged released source.
"""
from pathlib import Path
import hashlib
import importlib.util
import subprocess
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/Point-Cloud-Color-Constancy'
REVISION='ddc8770e04ef5509a621c5a7ee04e330b9204a93'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def identities():
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    assert revision==REVISION
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=no'],text=True).strip()
    extra=subprocess.check_output(['git','-C',str(SOURCE),'ls-files','--others','--exclude-standard'],text=True).splitlines()
    assert not any(name.endswith('.py') for name in extra),'Unexpected untracked Python source'
    names=['pointnet/model.py','pointnet/dataloader.py','utils_p.py','train_main.py']
    return {'repository':'https://github.com/xyxingx/Point-Cloud-Color-Constancy',
        'revision':revision,'files':{name:sha(SOURCE/name) for name in names}}

def load():
    sys.dont_write_bytecode=True
    identities()
    path=SOURCE/'pointnet/model.py';code=path.read_text(encoding='utf-8')
    before='x, trans = self.feat(x)';assert code.count(before)==1
    module=types.ModuleType('pccc_author_model');module.__file__=str(path)
    exec(compile(code.replace(before,'x, trans, _ = self.feat(x)'),str(path),'exec'),module.__dict__)
    def file_module(name,relative):
        spec=importlib.util.spec_from_file_location(name,SOURCE/relative)
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
    data=file_module('pccc_author_data','pointnet/dataloader.py')
    utils=file_module('pccc_author_utils','utils_p.py')
    return module,data,utils

def make_model(author,head):
    model=author.PCCC(k=3,sis=256,feature_transform=False)
    if head=='paper_weighted':model.conv4=torch.nn.Conv1d(128,4,1)
    else:assert head=='released_sum'
    return model.cpu()

def predict(model,points,head):
    local,_=model(points)
    if head=='released_sum':return local.sum(2)
    assert head=='paper_weighted' and local.shape[1]==4
    # Paper Eq.4: normalize certainty over points, then aggregate point RGB.
    weights=torch.softmax(local[:,3:4],dim=2)
    return (local[:,:3]*weights).sum(2)
