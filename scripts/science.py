"""Photometric transformations and depth metrics for a paired pilot experiment.

Depth is in meters. All methods use the same crop and reference mask. No scale
alignment is fitted to a prediction. Color changes operate in linear RGB.
"""
import numpy as np

CASTS={
    'clean':(1.,1.,1.),
    'warm_mild':(1.25,1.,.8),
    'warm_strong':(1.6,1.,.625),
    'cool_mild':(.8,1.,1.25),
    'cool_strong':(.625,1.,1.6),
    'green_mild':(.9,1.2,.9),
    'green_strong':(.75,1.6,.75),
}
METHODS=['none','gray_world','white_patch','gray_world_half']
LUMA=np.array([.2126,.7152,.0722],dtype=np.float32)

def srgb_to_linear(rgb):
    x=np.asarray(rgb,dtype=np.float32)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def linear_to_srgb(rgb):
    x=np.clip(np.asarray(rgb,dtype=np.float32),0,1)
    return np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)

def apply_gains(linear,gains):
    changed=linear*np.asarray(gains,dtype=np.float32)
    before=float(np.mean(linear@LUMA));after=float(np.mean(changed@LUMA))
    changed*=before/max(after,1e-8)
    clipped=float(np.mean(np.any((changed<0)|(changed>1),axis=2)))
    return np.clip(changed,0,1),clipped

def transform(rgb,condition,method):
    linear=srgb_to_linear(rgb)
    cast,clipped=apply_gains(linear,CASTS[condition])
    if method=='none':return linear_to_srgb(cast),dict(cast_clipped_fraction=clipped,correction_clipped_fraction=0.)
    if method in ('gray_world','gray_world_half'):
        illum=cast.mean(axis=(0,1),dtype=np.float64)
    elif method=='white_patch':
        illum=cast.max(axis=(0,1))
    else:raise ValueError(method)
    illum=np.maximum(illum,1e-6)
    gains=np.exp(np.mean(np.log(illum)))/illum
    if method=='gray_world_half':gains=np.sqrt(gains)
    corrected,cclip=apply_gains(cast,gains)
    return linear_to_srgb(corrected),dict(cast_clipped_fraction=clipped,correction_clipped_fraction=cclip,
                                         estimated_illuminant=illum.tolist(),correction_gains=gains.tolist())

def evaluate_depth(pred,gt):
    """NYU-style Eigen crop scaled from 480x640; 0.1-10m valid reference.

    The pilot averages per-image scores, rather than pooling all pixels.
    Predictions are clipped to [0.1,10]m, with invalid values rejected.
    """
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape:raise ValueError(f'Depth shapes differ: {pred.shape}, {gt.shape}')
    if not np.isfinite(pred).all():raise ValueError('Prediction contains nonfinite values')
    h,w=gt.shape;crop=np.zeros((h,w),dtype=bool)
    top,bottom=round(45*h/480),round(471*h/480)
    left,right=round(41*w/640),round(601*w/640)
    crop[top:bottom,left:right]=True
    valid=crop&np.isfinite(gt)&(gt>.1)&(gt<10.)
    if valid.sum()<100:raise ValueError('Insufficient valid depth pixels')
    p=np.clip(pred[valid],.1,10.);g=gt[valid];difference=p-g
    ratio=np.maximum(p/g,g/p)
    return dict(abs_rel=float(np.mean(np.abs(difference)/g)),rmse_m=float(np.sqrt(np.mean(difference**2))),
                mae_m=float(np.mean(np.abs(difference))),delta1=float(np.mean(ratio<1.25)),valid_pixels=int(valid.sum()))
