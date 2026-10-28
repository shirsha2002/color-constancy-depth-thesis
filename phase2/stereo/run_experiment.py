"""Small real-light DTU stereo experiment with calibrated structured-light GT.

Example calibration pair (wait for source files to finish downloading):
  python phase2/stereo/run_experiment.py --scenes 1 --refs 25 --sources 26 --lights 3,0 --out calibration_pair
Final fixed pilot:
  python phase2/stereo/run_experiment.py --scenes 1,6 --out main
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.io import loadmat

from geometry import camera,read_ply_xyz,zbuffer,visibility_mask,backproject,write_ply
from costs import METHODS,compute

ROOT=Path(__file__).resolve().parents[2]
PHASE=ROOT/'phase2'
DATA=PHASE/'data/dtu/source/SampleSet/MVS Data'
HERE=Path(__file__).resolve().parent


def csv_write(rows,path):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def input_image(scene,view,light,shape):
    p=DATA/f'Rectified/scan{scene}/rect_{view:03d}_{light}_r5000.png'
    im=cv2.imread(str(p))
    if im is None:raise FileNotFoundError(f'Original RGB not available yet: {p}')
    if im.shape[:2]!=(1200,1600):raise ValueError(f'Unexpected original resolution: {im.shape}')
    return cv2.resize(im[:,:,::-1],(shape[1],shape[0]),interpolation=cv2.INTER_AREA),p


def diagnostics(rgb,mask):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    score=cv2.Sobel(gray,cv2.CV_32F,1,0)**2+cv2.Sobel(gray,cv2.CV_32F,0,1)**2
    score[~mask]=-1
    h,w=mask.shape;points=[]
    for yi in range(3):
        for xi in range(4):
            y0,y1=yi*h//3,(yi+1)*h//3;x0,x1=xi*w//4,(xi+1)*w//4
            sub=score[y0:y1,x0:x1]
            if sub.max()<0:continue
            py,px=np.unravel_index(sub.argmax(),sub.shape);points.append((int(py+y0),int(px+x0)))
    return points


def save_geometry_figure(rgb,gt,mask,scene,view,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,3.9))
    ax[0].imshow(rgb);ax[0].contour(np.isfinite(gt),levels=[.5],colors=['#00e7be'],linewidths=.6)
    ax[0].set_title('RGB + projected STL support')
    d=ax[1].imshow(np.ma.masked_invalid(gt),cmap='magma_r',vmin=250,vmax=1200)
    ax[1].set_title('Z-buffered structured-light depth (mm)')
    ax[2].imshow(rgb);overlay=np.zeros((*mask.shape,4));overlay[mask]=[0,.8,.6,.45];ax[2].imshow(overlay)
    ax[2].set_title('Shared foreground + source visibility mask')
    for a in ax:a.axis('off')
    fig.colorbar(d,ax=ax[1],fraction=.035,pad=.02)
    fig.suptitle(f'DTU scene {scene} | supplied camera view {view} | no fitted geometry/scale')
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    cv2.setNumThreads(2)
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenes',default='1,6')
    ap.add_argument('--refs',default='25,26,35')
    ap.add_argument('--sources',default='',help='Override source views for a small development pair only')
    ap.add_argument('--lights',default='0,1,2,3,4,5,6')
    ap.add_argument('--out',default='main')
    ap.add_argument('--pixel-origin',type=int,choices=[0,1],default=None)
    args=ap.parse_args()
    scenes=[int(x) for x in args.scenes.split(',')];refs=[int(x) for x in args.refs.split(',')]
    lights=[int(x) for x in args.lights.split(',')]
    p=json.loads((HERE/'protocol.json').read_text())
    p['opencv_threads']=cv2.getNumThreads()
    if args.pixel_origin is not None:p['pixel_origin']=args.pixel_origin
    p['actual_scenes']=scenes;p['actual_reference_views']=refs;p['actual_source_lights']=lights
    p['actual_source_override']=[int(x) for x in args.sources.split(',')] if args.sources else None
    p['ground_plane_mask']='Original supplied Plane{scene}.mat, homogeneous plane value>0, evaluation only'
    if args.sources and 6 in scenes:raise ValueError('Source override reserved for development calibration')
    code=''.join((HERE/f).read_text() for f in ['geometry.py','costs.py','run_experiment.py'])
    p['code_sha256']=hashlib.sha256(code.encode()).hexdigest()
    canonical=json.dumps(p,sort_keys=True);fingerprint=hashlib.sha256(canonical.encode()).hexdigest()
    out=PHASE/'results/stereo'/args.out
    (out/'records').mkdir(parents=True,exist_ok=True);(out/'arrays').mkdir(exist_ok=True)
    (out/'figures').mkdir(exist_ok=True);(out/'pointclouds').mkdir(exist_ok=True)
    config=out/'protocol.json'
    if config.exists() and config.read_text()!=json.dumps(p,indent=2):
        raise ValueError('Existing output uses a different fixed protocol/code; choose a new --out')
    # The complete protocol is written before any matching or held-out scoring.
    config.write_text(json.dumps(p,indent=2),encoding='utf-8')
    shape=(p['height'],p['width']);depths=np.linspace(p['depth_min_mm'],p['depth_max_mm'],p['depth_planes'],dtype=np.float32)
    allrows=[];start=time.perf_counter()
    for scene in scenes:
        ply=DATA/f'Points/stl/stl{scene:03d}_total.ply'
        if not ply.exists():raise FileNotFoundError(ply)
        print(f'Loading original STL scene{scene}',flush=True);xyz=read_ply_xyz(ply)
        cams={v:camera(DATA/f'Calibration/cal18/pos_{v:03d}.txt',shape=shape,pixel_origin=p['pixel_origin']) for v in p['views_1based']}
        gt={v:zbuffer(xyz,cams[v]) for v in cams}
        plane=loadmat(DATA/f'ObsMask/Plane{scene}.mat')['P'].ravel()
        scene_meta={'scene':scene,'vertices':len(xyz),'units':'mm','views':{}}
        for ref in refs:
            src=p['actual_source_override'] or [v for v in p['views_1based'] if v!=ref]
            rcam=cams[ref];rgt=gt[ref]
            ref_rgb,ref_path=input_image(scene,ref,p['reference_light'],shape)
            X=backproject(rgt,rcam);foreground=np.isfinite(rgt)&((X@plane[:3]+plane[3])>0)
            ev=foreground.copy()
            for v in src:ev&=visibility_mask(rgt,gt[v],rcam,cams[v],p['source_visibility_tolerance_mm'])
            if ev.sum()<100:raise ValueError(f'Insufficient projected visible foreground: scene{scene} ref{ref}: {ev.sum()}')
            pts=diagnostics(ref_rgb,ev)
            save_geometry_figure(ref_rgb,rgt,ev,scene,ref,out/'figures'/f'geometry_scene{scene:03d}_view{ref:03d}.png')
            scene_meta['views'][str(ref)]={'gt_projected_pixels':int(np.isfinite(rgt).sum()),'gt_foreground_pixels':int(foreground.sum()),
                'shared_visible_foreground_pixels':int(ev.sum()),'gt_min_mm':float(np.nanmin(rgt)),'gt_max_mm':float(np.nanmax(rgt)),
                'camera_K':rcam['K'].tolist(),'camera_center_mm':rcam['C'].tolist(),'diagnostic_points_yx':pts}
            for light in lights:
                tag=f'scene{scene:03d}_ref{ref:03d}_src'+('-'.join(str(v) for v in src))+f'_light{light}'
                recfile=out/'records'/f'{tag}.json';arrfile=out/'arrays'/f'{tag}.npz'
                if recfile.exists() and arrfile.exists():
                    record=json.loads(recfile.read_text());allrows.extend(record['metrics']);print('cached '+tag,flush=True);continue
                source_rgbs=[];source_paths=[]
                for v in src:
                    im,path=input_image(scene,v,light,shape);source_rgbs.append(im);source_paths.append(str(path.relative_to(PHASE)))
                tick=time.perf_counter()
                result=compute(ref_rgb,source_rgbs,rcam,[cams[v] for v in src],depths,p['patch_width'],p['gain_ridge'],p['gain_bounds'],pts)
                seconds=time.perf_counter()-tick;mask=ev&result['support']
                if mask.sum()<100:raise ValueError('Insufficient common geometry support')
                finite=np.isfinite(result['depths'])
                if not all(np.array_equal(finite[0],f) for f in finite):raise ValueError('Methods have unequal output support')
                quant=np.min(np.abs(rgt[mask,None]-depths[None,:]),axis=1)
                metrics=[]
                for mi,m in enumerate(METHODS):
                    pred=result['depths'][mi];error=pred[mask]-rgt[mask]
                    if not np.isfinite(error).all():raise ValueError('Nonfinite prediction on shared scoring mask')
                    row={'scene':scene,'split':'development' if scene==p['development_scene'] else 'evaluation',
                        'reference_view':ref,'source_views':'-'.join(map(str,src)),'reference_light':p['reference_light'],'source_light':light,
                        'lighting':'matched_diffuse' if light==p['reference_light'] else 'mismatched','method':m,
                        'mae_mm':float(np.mean(np.abs(error))),'rmse_mm':float(np.sqrt(np.mean(error**2))),
                        'bad5_percent':float(100*np.mean(np.abs(error)>5)), 'bad10_percent':float(100*np.mean(np.abs(error)>10)),
                        'evaluation_pixels':int(mask.sum()),'coverage_percent':float(100*mask.sum()/max(int(foreground.sum()),1)),
                        'gt_in_search_percent':float(100*np.mean((rgt[mask]>=depths[0])&(rgt[mask]<=depths[-1]))),
                        'quantization_floor_mae_mm':float(quant.mean()),'all_method_shared_seconds':seconds}
                    metrics.append(row);allrows.append(row)
                    if light in (0,p['reference_light']) and m in ('rgb_ssd','diagonal_gain','channel_zncc'):
                        world=backproject(pred,rcam);write_ply(out/'pointclouds'/f'{tag}__{m}.ply',world[mask],ref_rgb[mask])
                np.savez_compressed(arrfile,depths=result['depths'],gt=rgt,mask=mask,foreground=foreground,ref_rgb=ref_rgb,
                    best_cost=result['best_cost'],curves=result['curves'],planes_mm=depths,points_yx=np.array(pts),
                    gain_priors=result['priors'],scalar_priors=result['scalar_priors'])
                record={'tag':tag,'protocol_sha256':fingerprint,'reference_image':str(ref_path.relative_to(PHASE)),
                    'source_images':source_paths,'seconds':seconds,'gain_priors':result['priors'].tolist(),
                    'scalar_priors':result['scalar_priors'].tolist(),'metrics':metrics}
                recfile.write_text(json.dumps(record,indent=2),encoding='utf-8')
                csv_write(allrows,out/'metrics.csv')
                print(f'{tag}: {seconds:.2f}s; '+', '.join(f'{m["method"]}={m["mae_mm"]:.2f}mm' for m in metrics),flush=True)
        (out/f'geometry_scene{scene:03d}.json').write_text(json.dumps(scene_meta,indent=2),encoding='utf-8')
    # Describe means only: the held-out pilot contains one scene.
    groups=[]
    for split in ['development','evaluation']:
        for lighting in ['matched_diffuse','mismatched']:
            for method in METHODS:
                rr=[r for r in allrows if r['split']==split and r['lighting']==lighting and r['method']==method]
                if rr:
                    groups.append({'split':split,'lighting':lighting,'method':method,'view_light_cases':len(rr),
                        'mae_mm':float(np.mean([r['mae_mm'] for r in rr])),'rmse_mm':float(np.mean([r['rmse_mm'] for r in rr])),
                        'bad5_percent':float(np.mean([r['bad5_percent'] for r in rr]))})
    csv_write(groups,out/'summary.csv')
    summary={'completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'protocol_sha256':fingerprint,
        'rows':len(allrows),'wall_seconds':time.perf_counter()-start,'summary':groups,
        'limitations':['One held-out scene; descriptive results only, no population confidence intervals',
            '320x240 plane sweep, finite depth planes and projected point samples, not full-resolution DTU benchmark',
            'GT source visibility and foreground plane define evaluation mask only; no GT enters inference',
            'DTU lights primarily change direction; gains model radiometric nuisance, no absolute illuminant recovery claim']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':main()
