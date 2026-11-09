"""Compare real PCCC forward/backward on CPU and Intel GPU, using train data."""
import argparse
import copy
import importlib.metadata as metadata
import json
import re
from pathlib import Path
import sys
import time
import numpy as np
import torch
from model import ROOT,load,make,forward,synchronize

HOME=ROOT/'phase5_pccc'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','xpu'],required=True);args=parser.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    output={'device_requested':args.device,'torch':torch.__version__,'python':sys.version,
        'xpu_available':torch.xpu.is_available(),'cuda_available':torch.cuda.is_available(),'deterministic_algorithms':True,
        'test_data_used':False,'timing':'Warm8trainingsteps, measure20batch16forward+backward steps, synchronized; no optimizer step in timed kernel'}
    try:
        if args.device=='xpu':
            assert torch.xpu.is_available(),'XPU is unavailable with the installed driver/runtime'
            output['gpu_name']=torch.xpu.get_device_name(0)
            output['gpu_properties']=re.sub(r'uuid=[^,]+','uuid=omitted',str(torch.xpu.get_device_properties(0)))
        author,data,utils=load();manifest=json.loads((HOME/'manifest.json').read_text())
        ids=manifest['folds']['0']['train_ids'][:16];by_id={r['id']:r for r in manifest['samples']}
        x=torch.from_numpy(np.stack([np.load(ROOT/by_id[k]['cloud']).astype(np.float32).T for k in ids]))
        y=torch.from_numpy(np.stack([np.load(ROOT/by_id[k]['label']).astype(np.float32) for k in ids]))
        results=[]
        for head in ['released','weighted']:
            torch.manual_seed(20261109);cpu=make(author,head,'cpu');cpu.eval()
            with torch.no_grad():reference=forward(cpu,x,head).clone()
            state=copy.deepcopy(cpu.state_dict());target=make(author,head,args.device);target.load_state_dict(state);target.eval()
            xx=x.to(args.device);yy=y.to(args.device)
            with torch.no_grad():actual=forward(target,xx,head).cpu()
            difference=float((actual-reference).abs().max());scale=float(reference.abs().max())
            assert torch.allclose(actual,reference,rtol=3e-3,atol=3e-4),(head,difference,scale)
            # Match the phase-four corrected author network on CPU exactly.
            import upstream
            older,_,_=upstream.load();original=upstream.make_model(older,'released_sum' if head=='released' else 'paper_weighted')
            original.load_state_dict(state);original.eval()
            with torch.no_grad():old=upstream.predict(original,x,'released_sum' if head=='released' else 'paper_weighted')
            assert torch.equal(old,reference),'Portable identity fix changed CPU numerics'
            # The redundant-pool bypass must also preserve CPU gradients.
            original.train();cpu.train();original.zero_grad(set_to_none=True);cpu.zero_grad(set_to_none=True)
            utils.get_angular_loss(upstream.predict(original,x,'released_sum' if head=='released' else 'paper_weighted'),y).backward()
            utils.get_angular_loss(forward(cpu,x,head),y).backward()
            old_grads=[p.grad for p in original.parameters()];new_grads=[p.grad for p in cpu.parameters()]
            grad_difference=max(float((a-b).abs().max()) for a,b in zip(old_grads,new_grads))
            assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-7) for a,b in zip(old_grads,new_grads)),'Identity pooling changed CPU gradients'
            target.train()
            for _ in range(8):
                target.zero_grad(set_to_none=True);loss=utils.get_angular_loss(forward(target,xx,head),yy);loss.backward()
            synchronize(args.device);start=time.perf_counter()
            for _ in range(20):
                target.zero_grad(set_to_none=True);loss=utils.get_angular_loss(forward(target,xx,head),yy);loss.backward()
            synchronize(args.device);seconds=(time.perf_counter()-start)/20
            assert torch.isfinite(loss) and torch.isfinite(target.conv4.weight.grad).all()
            # Exercise a real Adam update, not just inference.
            optimizer=torch.optim.Adam(target.parameters(),lr=.0003);optimizer.step();synchronize(args.device)
            results.append({'head':head,'batch16_forward_backward_seconds':seconds,
                'cpu_reference_max_component_difference':difference,'reference_max_abs':scale,
                'finite_angular_backward_and_adam':True,'portable_fix_cpu_bitwise_matches_previous':True})
            results[-1]['pool_bypass_cpu_gradient_max_difference']=grad_difference
            results[-1]['pool_bypass_cpu_gradients_bitwise_identical']=all(torch.equal(a,b) for a,b in zip(old_grads,new_grads))
        output['status']='PASS';output['measurements']=results
    except Exception as e:
        output['status']='FAILED';output['error_type']=type(e).__name__;output['error']=str(e)
    path=HOME/f'benchmark_{args.device}.json';path.write_text(json.dumps(output,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(output,indent=2),flush=True)
    if output['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
