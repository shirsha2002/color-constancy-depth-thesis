"""Numerical validation independent of depth-score reporting.

Checks actual supplied cameras and saved development-pair cost curves. The
gain objectives are independently solved by a generic bounded optimizer;
correlations use numpy.corrcoef rather than box-filter moment identities.
"""
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from geometry import camera,backproject,project,warps
from costs import linear_rgb,LUMA,METHODS
from run_experiment import DATA,PHASE,input_image


def correlation(x,y):
    x=x.ravel().astype(np.float64);y=y.ravel().astype(np.float64)
    if np.std(x)*np.std(y)<=1e-6:return 0.
    return float(np.corrcoef(x,y)[0,1])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pair-run',default='calibration_pair_v2');args=ap.parse_args()
    report={'camera_roundtrips':[],'development_cost_checks':[]}
    shape=(240,320)
    for view in [25,26,35]:
        c=camera(DATA/f'Calibration/cal18/pos_{view:03d}.txt',shape=shape)
        y,x=np.mgrid[:shape[0],:shape[1]];target=np.stack([x.ravel(),y.ravel()],axis=1)
        for depth in [250,500,1200]:
            X=backproject(np.full(shape,depth,dtype=np.float64),c)
            uv,z=project(X.reshape(-1,3),c);pixel_error=float(np.abs(uv-target).max());z_error=float(np.abs(z-depth).max())
            if pixel_error>1e-8 or z_error>1e-8:raise ValueError('Camera roundtrip failed')
            mx,my,valid=warps(c,c,depth);identity=float(max(np.abs(mx-x).max(),np.abs(my-y).max()))
            if identity>1e-5:raise ValueError('Identity plane homography failed')
            report['camera_roundtrips'].append({'view':view,'depth_mm':depth,'max_pixel_error':pixel_error,'max_depth_error_mm':z_error,'identity_warp_error_pixels':identity})
    path=PHASE/f'results/stereo/{args.pair_run}/arrays/scene001_ref025_src26_light0.npz'
    a=np.load(path);ref,_=input_image(1,25,3,shape);source,_=input_image(1,26,0,shape)
    r=linear_rgb(ref).astype(np.float64);s=linear_rgb(source).astype(np.float64)
    rc=camera(DATA/'Calibration/cal18/pos_025.txt',shape=shape);sc=camera(DATA/'Calibration/cal18/pos_026.txt',shape=shape)
    prior=np.clip(r.mean(axis=(0,1))/s.mean(axis=(0,1)),.25,4);scalar_prior=float(np.clip(r.mean()/s.mean(),.25,4))
    for pi,(py,px) in enumerate(a['points_yx'][:3]):
        for di in [96,128,160]:
            expected=a['curves'][:,di,pi]
            if not np.isfinite(expected).all():continue
            mx,my,valid=warps(rc,sc,float(a['planes_mm'][di]));sw=cv2.remap(s,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            y=r[py-2:py+3,px-2:px+3].astype(np.float64);x=sw[py-2:py+3,px-2:px+3].astype(np.float64)
            if x.shape!=(5,5,3):continue
            def f(g):return np.mean((y-g*x)**2)+.002*(g-scalar_prior)**2
            scalar=minimize_scalar(f,bounds=(.25,4),method='bounded',options={'xatol':1e-12}).fun
            diag=[]
            for ch in range(3):
                fun=lambda g:np.mean((y[:,:,ch]-g*x[:,:,ch])**2)+.002*(g-prior[ch])**2
                diag.append(minimize_scalar(fun,bounds=(.25,4),method='bounded',options={'xatol':1e-12}).fun)
            actual=np.array([np.mean((y-x)**2),scalar,np.mean(diag),1-correlation(y@LUMA,x@LUMA),np.mean([1-correlation(y[:,:,ch],x[:,:,ch]) for ch in range(3)])])
            delta=np.abs(actual-expected)
            if np.any(delta>2e-3):raise ValueError(f'Cost mismatch {py,px,di}: {delta}')
            report['development_cost_checks'].append({'pixel_yx':[int(py),int(px)],'plane_mm':float(a['planes_mm'][di]),
                'maximum_absolute_cost_difference':float(delta.max()),'per_method_absolute_difference':dict(zip(METHODS,map(float,delta)))})
    if not report['development_cost_checks']:raise ValueError('No valid independent cost checks')
    report['status']='passed';report['scope']='Algebra and cost validation; external RGB/STL support overlay separately inspected. No calibration scale fitted.'
    report['pair_run']=args.pair_run
    out=PHASE/'results/stereo/method_validation.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(out)


if __name__=='__main__':main()
