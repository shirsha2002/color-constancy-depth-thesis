"""Complement strict CPU/XPU gradient check with same-device repeatability.

Criteria set before execution: finite Adam updates, exact repeat on the same
XPU, and initial CPU/XPU angular prediction disagreement below 0.02 degrees.
This does not erase or relax the earlier failed gradient tolerance check.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from data import ROOT, MANIFESTS, split, sha, damage
from model import make_model, objective

HOME=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.xpu.is_available()
    m=json.loads((ROOT/MANIFESTS['ETH3D']).read_text())
    records={r['id']:r for r in m['samples']}; ids=split(m,0)['train'][:16]
    rng=np.random.default_rng(20261116); arrays=[[],[],[]]
    for sid in ids:
        r=records[sid]
        for field in ['cloud','label']: assert sha(ROOT/r[field])==r[field+'_sha256']
        x=np.load(ROOT/r['cloud']).astype(np.float32)
        for dest,item in zip(arrays,[x.T,damage(x,'noise30',rng).T,np.load(ROOT/r['label']).astype(np.float32)]): dest.append(item)
    cpu=[torch.from_numpy(np.stack(a)) for a in arrays]
    inputs=[a.to('xpu') for a in cpu]
    torch.manual_seed(20261116)
    original=make_model('pccc').train(); initial=copy.deepcopy(original.state_dict())
    cp=original(cpu[0])['illuminant'].detach().numpy().astype(float)
    outcomes=[]; losses=[]; first=None
    for repeat in range(2):
        torch.xpu.manual_seed_all(20261116)
        net=make_model('pccc').to('xpu').train(); net.load_state_dict(initial)
        pred=net(inputs[0])['illuminant'].detach().cpu().numpy().astype(float)
        if first is None:first=pred
        opt=torch.optim.Adam(net.parameters(),lr=.0003); history=[]
        for step in range(10):
            opt.zero_grad(set_to_none=True)
            loss=objective(net(inputs[0]),net(inputs[1]),inputs[2]); assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            opt.step(); history.append(float(loss.detach().cpu()))
        assert all(torch.isfinite(p).all() for p in net.parameters())
        outcomes.append({n:v.cpu().clone() for n,v in net.state_dict().items()}); losses.append(history)
    assert losses[0]==losses[1]
    for n in outcomes[0]: assert torch.equal(outcomes[0][n],outcomes[1][n]),n
    a=cp/np.linalg.norm(cp,axis=1,keepdims=True); b=first/np.linalg.norm(first,axis=1,keepdims=True)
    delta=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),(a*b).sum(1)))
    assert float(delta.max()) < .02
    result=dict(status='PASS_SAME_XPU_REPEATABILITY', ten_finite_updates_per_repeat=True,
                exact_repeated_losses=True, exact_repeated_final_state=True,
                cpu_xpu_initial_prediction_max_angle_difference_deg=float(delta.max()),
                earlier_strict_cpu_xpu_gradient_comparison='FAILED; retained; not waived or claimed to pass',
                interpretation='Backend is operational and repeatable; all pilot methods must use XPU. Cross-backend gradient identity is not established.',
                training_ids=ids, heldout_used=False, model_sha256=sha(HOME/'model.py'),
                checker_sha256=sha(Path(__file__)), paid_compute_launched=False)
    (HOME/'pccc_xpu_repeatability.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
