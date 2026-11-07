"""Loss-matched controls motivated only by NYU validation, before test inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import study
import spatial_study
from spatial_model import SpatialGeometryAdapter

CONTROLS={
    'spatial_content_noeq':('spatial_content',0.,.01),
    'spatial_flat_noeq':('spatial_flat',0.,.01),
    'spatial_noeq_no_graph':('spatial_geometry',0.,0.),
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate'])
    parser.add_argument('--input-size',type=int,default=224);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--steps',type=int,default=600);parser.add_argument('--out',type=Path,default=study.ROOT/'phase3/results/spatial_controls')
    args=parser.parse_args();torch.set_num_interop_threads(1);spatial_study.VARIANTS=CONTROLS
    cfg=spatial_study.configuration(args);cfg.pop('configuration_sha256')
    cfg['primary_method']=None;cfg['role']='Additional loss-matched controls, motivated by validation only'
    cfg['sources'][Path(__file__).relative_to(study.ROOT).as_posix()]=study.sha(Path(__file__))
    cfg['configuration_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==json.loads(json.dumps(cfg))
    else:study.dump(out/'config.json',cfg)
    for relative,expected in {**cfg['sources'],**cfg['manifests']}.items():assert study.sha(study.ROOT/relative)==expected
    model=study.frozen_model(args.threads);data=study.inputs();store=study.Store(model,cfg)
    if args.stage=='train':spatial_study.train(model,data,store,cfg,out)
    else:
        study.GeometryChromaticAdapter=SpatialGeometryAdapter
        study.VARIANTS={k:[v[0],v[1]] for k,v in CONTROLS.items()}
        study.evaluate(model,data,store,cfg,out)

if __name__=='__main__':main()
