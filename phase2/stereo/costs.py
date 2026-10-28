"""Classical matching costs with shared warping and geometric support.

The gain costs analytically minimize a regularized photometric residual for
each geometric depth hypothesis. They are known-model engineering baselines,
not a claim of a new color-constancy algorithm.
"""
import cv2
import numpy as np
from geometry import warps

METHODS=['rgb_ssd','scalar_gain','diagonal_gain','gray_zncc','channel_zncc']
LUMA=np.array([.2126,.7152,.0722],dtype=np.float32)


def linear_rgb(im):
    x=np.asarray(im,dtype=np.float32)/255.
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)


def box(x,k):
    return cv2.boxFilter(x,-1,(k,k),normalize=True,borderType=cv2.BORDER_CONSTANT)


def compute(ref_rgb,source_rgbs,ref_cam,source_cams,depths,window=5,ridge=.002,bounds=(.25,4),points=None):
    """Stream plane costs; keep only winners and small diagnostic cost curves."""
    # Double precision prevents cancellation of tiny local variance near white
    # or black patches; otherwise established ZNCC can be weakened numerically.
    r=linear_rgb(ref_rgb).astype(np.float64);ss=[linear_rgb(s).astype(np.float64) for s in source_rgbs]
    h,w=r.shape[:2];M=len(METHODS);best=np.full((M,h,w),np.inf,np.float32)
    second=best.copy();depthmap=np.full_like(best,np.nan);support=np.zeros((h,w),bool)
    mr=box(r,window);rr=box(r*r,window);vr=np.maximum(rr-mr*mr,0)
    rg=r@LUMA;mrg=box(rg,window);rrg=box(rg*rg,window);vrg=np.maximum(rrg-mrg*mrg,0)
    priors=[np.clip(r.mean(axis=(0,1))/np.maximum(s.mean(axis=(0,1)),1e-6),*bounds) for s in ss]
    scalar_priors=[float(np.clip(r.mean()/max(float(s.mean()),1e-6),*bounds)) for s in ss]
    curves=np.full((M,len(depths),len(points or [])),np.nan,dtype=np.float32)
    for di,z in enumerate(depths):
        allcost=np.zeros((M,h,w),np.float32);valid=np.ones((h,w),bool)
        for s,cam,prior,scalar_prior in zip(ss,source_cams,priors,scalar_priors):
            x,y,g=warps(ref_cam,cam,float(z));warped=cv2.remap(s,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            # Every pixel in the patch must have a valid bilinear source sample.
            valid&=box(g.astype(np.float32),window)>.99999
            ms=box(warped,window);s2=box(warped*warped,window);rs=box(r*warped,window)
            raw=np.maximum((rr-2*rs+s2).mean(axis=2),0)
            a=np.clip((rs.mean(axis=2)+ridge*scalar_prior)/(s2.mean(axis=2)+ridge),*bounds)
            scalar=np.maximum(rr.mean(axis=2)-2*a*rs.mean(axis=2)+a*a*s2.mean(axis=2)+ridge*(a-scalar_prior)**2,0)
            ac=np.clip((rs+ridge*prior)/(s2+ridge),*bounds)
            diag=np.maximum((rr-2*ac*rs+ac*ac*s2+ridge*(ac-prior)**2).mean(axis=2),0)
            vs=np.maximum(s2-ms*ms,0);cov=rs-mr*ms
            den=np.sqrt(vr*vs)
            corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-6)
            channel=(1-np.clip(corr,-1,1)).mean(axis=2)
            sg=warped@LUMA;msg=box(sg,window);ssg=box(sg*sg,window);rsg=box(rg*sg,window)
            vsg=np.maximum(ssg-msg*msg,0);covg=rsg-mrg*msg;deng=np.sqrt(vrg*vsg)
            corr=np.divide(covg,deng,out=np.zeros_like(covg),where=deng>1e-6)
            gray=1-np.clip(corr,-1,1)
            allcost+=np.stack([raw,scalar,diag,gray,channel])
        allcost/=len(ss);allcost[:,~valid]=np.inf;support|=valid
        better=allcost<best
        second=np.where(better,best,np.minimum(second,allcost))
        best=np.where(better,allcost,best);depthmap=np.where(better,z,depthmap)
        if points:
            for pi,(py,px) in enumerate(points):curves[:,di,pi]=allcost[:,py,px]
    return {'depths':depthmap,'best_cost':best,'second_cost':second,'support':support,
            'curves':curves,'priors':np.array(priors),'scalar_priors':np.array(scalar_priors)}
