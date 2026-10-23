"""Apply the implemented pipeline to an indoor RGB image without reference depth.

Example: .venv/Scripts/python scripts/predict_image.py --image room.jpg
Output depth arrays are estimates in meters, not measured ground truth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import ROOT,create_model
from science import transform,METHODS,CASTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,default='gray_world_half')
    parser.add_argument('--cast',choices=list(CASTS),default='clean')
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/demo')
    args=parser.parse_args()
    rgb=np.asarray(Image.open(args.image).convert('RGB'),dtype=np.float32)/255
    processed,meta=transform(rgb,args.cast,args.method)
    model=create_model(8)
    import torch
    with torch.inference_mode():
        prediction=model.infer_image(np.round(processed[:,:,::-1]*255).astype(np.uint8),input_size=518)
    args.outdir.mkdir(parents=True,exist_ok=True)
    np.save(args.outdir/'depth_meters.npy',prediction.astype(np.float32))
    Image.fromarray(np.round(processed*255).astype(np.uint8)).save(args.outdir/'processed_rgb.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].imshow(rgb);axes[0].set_title('Original RGB')
    axes[1].imshow(processed);axes[1].set_title(args.method.replace('_',' ').title())
    im=axes[2].imshow(prediction,vmin=.1,vmax=10,cmap='magma');axes[2].set_title('Estimated depth (meters)')
    for ax in axes:ax.axis('off')
    fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04,label='m')
    fig.tight_layout();fig.savefig(args.outdir/'comparison.png',dpi=160);plt.close(fig)
    record=dict(input=str(args.image),method=args.method,cast=args.cast,depth_units='meters',
                reference_depth_available=False,accuracy_measured=False,transformation=meta)
    (args.outdir/'prediction.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Saved estimated depth and comparison to {args.outdir}')

if __name__=='__main__':main()
