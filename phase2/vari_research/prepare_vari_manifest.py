"""Preserve16bit rendered depth and audit the officialPIL8 compatibility path."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import numpy as np
import tifffile
from PIL import Image

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
DEST=HERE.parent/'data/vari'
def rel(p):return p.relative_to(PROJECT).as_posix()
def rawpath(n):return DEST/'raw'/Path(*PurePosixPath(n).parts)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit-groups',type=int,default=0)
    ap.add_argument('--output',default='manifest.json');args=ap.parse_args()
    declared=json.loads((DEST/'exploratory_split_groups.json').read_text())
    groups=declared['groups'][:args.limit_groups] if args.limit_groups else declared['groups']
    images=[];audit=[]
    for g in groups:
        split='training' if g['split']=='train' else g['split']
        paths=[rawpath(g['depth']),*[rawpath(n) for n in g['rgb'].values()]]
        missing=[str(p) for p in paths if not p.exists()]
        if missing:raise FileNotFoundError(f'Group{g["view_group"]} incomplete: {missing[:3]}')
        raw=paths[0];a=tifffile.imread(raw)
        if a.dtype!=np.uint16:raise ValueError('Primary decoder requires actualuint16TIFF')
        if a.ndim==3:
            equal=bool(np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,0],a[...,2]))
            if not equal:raise ValueError('DepthRGBchannelsareunequal; investigatebeforeconversion')
            scalar=a[...,0]
        elif a.ndim==2:equal=True;scalar=a
        else:raise ValueError('Unexpecteddepthshape')
        depth=((1-scalar.astype(np.float32)/65535)*10).astype(np.float32)
        with Image.open(raw) as im:
            pilmode=im.mode
            legacy=((1-np.asarray(im.convert('L'),np.float32)/255)*10).astype(np.float32)
            bits=list(im.tag_v2.get(258,[]))
        primary=DEST/'depth_m16'/f'{g["view_group"]}.npy'
        compat=DEST/'depth_m_author8'/f'{g["view_group"]}.npy'
        primary.parent.mkdir(exist_ok=True);compat.parent.mkdir(exist_ok=True)
        np.save(primary,depth);np.save(compat,legacy)
        difference=np.abs(depth-legacy)
        rec={'view_group':g['view_group'],'scene':g['scene'],'split':split,
             'raw_depth':rel(raw),'raw_sha256':sha(raw),'stored_dtype':str(a.dtype),
             'stored_shape':list(a.shape),'stored_min':int(a.min()),'stored_max':int(a.max()),
             'bits_per_sample':bits,'channels_bitidentical':equal,'PIL_mode':pilmode,
             'depth':rel(primary),'depth_sha256':sha(primary),'legacy_depth':rel(compat),
             'legacy_depth_sha256':sha(compat),'primary_range_m':[float(depth.min()),float(depth.max())],
             'primary_step_m':10/65535,'legacy_step_m':10/255,
             'primary_legacy_mean_absolute_difference_m':float(difference.mean()),
             'primary_legacy_max_absolute_difference_m':float(difference.max()),
             'lighting_reference_equality':'All11lightingimagesinthisgroupreferenceoneidenticalrawTIFFanddecodedNPY'}
        audit.append(rec)
        for light,name in g['rgb'].items():
            rgb=rawpath(name)
            with Image.open(rgb) as im:w,h=im.size;mode=im.mode
            if depth.shape!=(h,w):raise ValueError('RGBanddepthmisaligneddimensions')
            images.append({'id':g['view_group']+'_'+light,'scene':g['scene'],'view_group':g['view_group'],
                'camera':g['camera'],'view':g['view'],'lighting':light,'split':split,'rgb':rel(rgb),
                'depth':rel(primary),'legacy_depth':rel(compat),'width':w,'height':h,'rgb_mode':mode,
                'rgb_sha256':sha(rgb),'raw_depth_sha256':rec['raw_sha256'],'depth_sha256':rec['depth_sha256']})
    output={'dataset':'Vari releasedTEST assets, custom exploratory scene-separated split',
        'provenance':{'homepage':'https://github.com/zzt76/dca','source_commit':'2c8fde439bc94b56783b267198b3fe7e00b23fcf',
        'download_url':'https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87&download=1',
        'access':'Anonymous ordinaryHTTPbyteranges; every extractedZIPmemberCRCverified',
        'created_at_utc':datetime.now(timezone.utc).isoformat()},
        'selection_seed':20261028,'groups_by_split':dict(Counter(r['split'] for r in audit)),
        'images_by_split':dict(Counter(r['split'] for r in images)),
        'scene_assignment':{'212':'training+validation, disjointviewgroups','485':'evaluationonly'},
        'depth_units':'meters by invertednormalized10m convention used by authors',
        'depth_decode':{'primary':'(1-uint16/65535)*10, preserveoriginalTIFFprecision',
            'legacy':'AuthorsPILconvertL8bit then(1-gray/255)*10',
            'disclosure':'Primary16bitdecodeis an explicit exploratory refinement, not exact officialbenchmarkreplication',
            'primary_step_m':10/65535,'legacy_step_m':10/255},
        'warnings':['Only2originalrooms; camera views and11lightingvariants are correlated.',
            'Training+validation share room212; validation is viewheldout, not roomheldout.',
            'Room485 provides a single unseenroom, not broad scene generalization.',
            'CustomsplitofreleasedTESTassets; noofficialVari trainingdatarelease was available.'],
        'images':images}
    target=DEST/args.output
    target.write_text(json.dumps(output,indent=2),encoding='utf8')
    auditpath=DEST/('first_group_depth_audit.json' if args.limit_groups else 'depth_decode_audit.json')
    auditpath.write_text(json.dumps({'groups':audit,
        'mean_group_mean_absolute_difference_m':float(np.mean([r['primary_legacy_mean_absolute_difference_m'] for r in audit])),
        'max_absolute_difference_m':max(r['primary_legacy_max_absolute_difference_m'] for r in audit),
        'all_lighting_reference_hashes_equal':all(len({im['depth_sha256'] for im in images if im['view_group']==g['view_group']})==1 for g in audit)},indent=2),encoding='utf8')
    print(json.dumps({'manifest':str(target),'groups':output['groups_by_split'],'images':output['images_by_split'],
        'first_image':images[0],'depth_audit':audit[0]},indent=2))
if __name__=='__main__':main()
