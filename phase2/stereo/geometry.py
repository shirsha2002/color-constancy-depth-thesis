"""Calibrated DTU geometry; all world, translation and depth lengths are mm."""
from pathlib import Path
import cv2
import numpy as np


def read_ply_xyz(path):
    """Read original DTU ASCII/binary PLY vertex XYZ without geometry libraries."""
    typ={'char':'i1','uchar':'u1','int8':'i1','uint8':'u1','short':'i2','ushort':'u2',
         'int16':'i2','uint16':'u2','int':'i4','uint':'u4','int32':'i4','uint32':'u4',
         'float':'f4','float32':'f4','double':'f8','float64':'f8'}
    with Path(path).open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError('Not a PLY')
        fmt=None;n=None;props=[];vertex=False
        while True:
            line=f.readline()
            if not line:raise ValueError('Truncated PLY header')
            a=line.decode('ascii').strip().split()
            if not a:continue
            if a[0]=='format':fmt=a[1]
            elif a[0]=='element':
                vertex=a[1]=='vertex'
                if vertex:n=int(a[2])
            elif a[0]=='property' and vertex:
                if a[1]=='list':raise ValueError('List property in vertex is unsupported')
                props.append((a[2],typ[a[1]]))
            elif a[0]=='end_header':break
        if n is None or fmt is None:raise ValueError('Missing PLY vertices/format')
        if fmt=='ascii':
            x=np.loadtxt(f,max_rows=n,dtype=np.float32,ndmin=2)
            names=[p[0] for p in props];xyz=x[:,[names.index(c) for c in ['x','y','z']]]
        else:
            endian='<' if fmt=='binary_little_endian' else '>'
            dtype=np.dtype([(name,endian+t) for name,t in props])
            x=np.fromfile(f,dtype=dtype,count=n)
            if len(x)!=n:raise ValueError('Truncated PLY vertices')
            xyz=np.stack([x[c] for c in ['x','y','z']],axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def camera(path,shape=(240,320),original_shape=(1200,1600),pixel_origin=0):
    P=np.loadtxt(path,dtype=np.float64)
    if P.shape!=(3,4):raise ValueError(f'Expected 3x4 P: {path}')
    K,R,Ch,*_=cv2.decomposeProjectionMatrix(P)
    K/=K[2,2];C=(Ch[:3]/Ch[3]).ravel();t=-R@C
    Pe=K@np.column_stack([R,t])
    sc=float(np.sum(P*Pe)/np.sum(Pe*Pe))
    if np.linalg.norm(P-sc*Pe)/np.linalg.norm(P)>1e-7:raise ValueError('P decomposition failed')
    if abs(np.linalg.det(R)-1)>1e-6:raise ValueError('Improper camera rotation')
    h,w=shape;oh,ow=original_shape;sx,sy=w/ow,h/oh
    # Pixel-center convention matches OpenCV resize. Origin is selected using
    # calibration scene only, never fitted to held-out scene geometry.
    S=np.array([[sx,0,sx*(.5-pixel_origin)-.5],
                [0,sy,sy*(.5-pixel_origin)-.5],[0,0,1]],dtype=np.float64)
    return {'K':S@K,'R':R,'t':t,'C':C,'P':S@P,'original_P':P,
            'shape':shape,'pixel_origin':pixel_origin,'scale':[sx,sy]}


def project(xyz,cam):
    xc=np.asarray(xyz)@cam['R'].T+cam['t']
    q=xc@cam['K'].T
    uv=q[:,:2]/q[:,2:3]
    return uv,xc[:,2]


def zbuffer(xyz,cam):
    h,w=cam['shape'];uv,z=project(xyz,cam)
    good=np.isfinite(uv).all(axis=1)&np.isfinite(z)&(z>0)
    uv=uv[good];z=z[good]
    ij=np.floor(uv+.5).astype(np.int64)
    good=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    ij=ij[good];z=z[good]
    flat=np.full(h*w,np.inf,dtype=np.float32)
    np.minimum.at(flat,ij[:,1]*w+ij[:,0],z.astype(np.float32))
    depth=flat.reshape(h,w);depth[~np.isfinite(depth)]=np.nan
    return depth


def backproject(depth,cam):
    h,w=depth.shape;xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    pix=np.stack([xx,yy,np.ones_like(xx)],axis=-1).astype(np.float64)
    rays=pix@np.linalg.inv(cam['K']).T
    xc=rays*np.asarray(depth)[...,None]
    return (xc-cam['t'])@cam['R']


def warps(ref,source,z):
    h,w=ref['shape'];xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    pix=np.stack([xx,yy,np.ones_like(xx)],axis=-1).astype(np.float64)
    X=backproject(np.full((h,w),z),ref)
    xc=X@source['R'].T+source['t'];q=xc@source['K'].T
    x=(q[...,0]/q[...,2]).astype(np.float32);y=(q[...,1]/q[...,2]).astype(np.float32)
    good=(xc[...,2]>0)&np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<w-1)&(y>=0)&(y<h-1)
    return x,y,good


def visibility_mask(ref_gt,source_gt,ref,source,tolerance_mm=5.):
    X=backproject(ref_gt,ref);uv,z=project(X.reshape(-1,3),source)
    h,w=source_gt.shape
    good=np.isfinite(uv).all(axis=1)&np.isfinite(z)&(z>0)
    ij=np.zeros_like(uv,dtype=np.int64)
    ij[good]=np.floor(uv[good]+.5).astype(np.int64)
    good&=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    sg=np.full(len(z),np.nan,dtype=np.float32)
    sg[good]=source_gt[ij[good,1],ij[good,0]]
    good&=np.isfinite(sg)&(np.abs(z-sg)<tolerance_mm)
    return good.reshape(ref_gt.shape)&np.isfinite(ref_gt)


def write_ply(path,xyz,rgb=None):
    p=np.asarray(xyz,dtype=np.float32).reshape(-1,3)
    good=np.isfinite(p).all(axis=1);p=p[good]
    if rgb is None:rgb=np.full((len(good),3),160,dtype=np.uint8)
    rgb=np.asarray(rgb,dtype=np.uint8).reshape(-1,3)[good]
    a=np.empty(len(p),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(['x','y','z']):a[k]=p[:,i]
    for i,k in enumerate(['red','green','blue']):a[k]=rgb[:,i]
    with Path(path).open('wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(p)}\n'
                 'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        a.tofile(f)
