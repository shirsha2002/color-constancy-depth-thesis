"""Audit the complete native ReMID release and map RGBs to shared metric depth.

No model predictions or error scores are read. ReMID is locked as an external
evaluation dataset; its transport shard IDs are never treated as model folds.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'phase3/data/remid';RAW=OUT/'raw'

def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):d.update(chunk)
 return d.hexdigest()

def rel(path):return path.relative_to(ROOT).as_posix()

def main():
 acquisition_path=OUT/'acquisition_manifest.json'
 acquisition=json.loads(acquisition_path.read_text(encoding='utf-8'))
 assert len(acquisition['assets'])==6
 for entry in acquisition['extracted_files']:
  path=ROOT/entry['path'];assert path.stat().st_size==entry['bytes'];assert sha(path)==entry['sha256']
 images=[];captures=[];depth_hash_captures=defaultdict(list)
 for meta_path in sorted(RAW.rglob('meta.json')):
  meta=json.loads(meta_path.read_text(encoding='utf-8'));folder=meta_path.parent
  shard=folder.relative_to(RAW).parts[0];category=meta['category'];scene_id=meta['scene_id']
  capture=f'{shard}/{category}/{scene_id}';conservative=f'{category}/{scene_id}'
  assert meta['depth']['unit']=='millimeter'
  assert meta['depth']['invalid_value']==0
  assert meta['camera']['depth_alignment_to_color']['enabled'] is True
  depth_path=folder/meta['depth']['raw_file'];depth=np.load(depth_path,allow_pickle=False)
  assert depth.shape==tuple(meta['depth']['shape']) and depth.dtype==np.float32
  valid=np.isfinite(depth)&(depth>0);v=depth[valid]
  assert len(v)>0
  depth_sha=sha(depth_path);depth_hash_captures[depth_sha].append(capture)
  rgb_paths=sorted((folder/meta['rgb']['directory']).glob('*.png'))
  assert len(rgb_paths)==meta['rgb']['n_images']
  capture_record={'capture_id':capture,'conservative_group':conservative,'category':category,'source_scene_id':scene_id,'transport_shard':shard,'rgb_images':len(rgb_paths),'depth':rel(depth_path),'depth_sha256':depth_sha,'meta':rel(meta_path),'meta_sha256':sha(meta_path),'depth_shape':list(depth.shape),'depth_dtype':str(depth.dtype),'depth_units':'millimeters','raw_positive_valid_fraction':float(valid.mean()),'positive_depth_quantiles_mm':dict(zip(['min','p01','median','p99','max'],map(float,np.quantile(v,[0,.01,.5,.99,1])))),'positive_gt_above_10m_pixels':int(np.count_nonzero(v>=10000)),'color_intrinsics':meta['camera']['color_intrinsics'],'alignment':meta['camera']['depth_alignment_to_color']}
  captures.append(capture_record)
  for rgb_path in rgb_paths:
   with Image.open(rgb_path) as rgb:
    width,height=rgb.size;assert rgb.mode=='RGB';assert (height,width)==depth.shape;rgb.verify()
   images.append({'id':f'remid_{shard}_{category}_{scene_id}_{rgb_path.stem}','scene':conservative,'source_scene_id':scene_id,'category':category,'transport_shard':shard,'view_group':capture,'conservative_group':conservative,'lighting':rgb_path.stem,'split':'evaluation','rgb':rel(rgb_path),'depth':rel(depth_path),'depth_units':'millimeters','depth_scale_to_meters':.001,'rgb_sha256':sha(rgb_path),'depth_sha256':depth_sha,'meta':rel(meta_path),'meta_sha256':capture_record['meta_sha256'],'width':width,'height':height})
 hashes=defaultdict(list)
 for record in images:hashes[record['rgb_sha256']].append(record['id'])
 duplicate_rgb=[{'sha256':k,'image_ids':v} for k,v in hashes.items() if len(v)>1]
 duplicate_depth=[{'sha256':k,'capture_ids':v} for k,v in depth_hash_captures.items() if len(v)>1]
 manifest={'dataset':'ReMID real multi-illumination RGB-D, complete author-published five-shard release','paper':'LumiDepth: Stable Monocular Depth in Multi-illumination Scenes (ECCV2026)','source_paper':'https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9#Fn1','public_folder':'https://drive.google.com/drive/folders/1-3txE3KiFiscACxs1Xch8UO1Ax82VBNg','acquisition_manifest_sha256':sha(acquisition_path),'role':'external evaluation only; never used for training, hyperparameter selection or checkpoint selection','images_by_split':{'evaluation':len(images)},'capture_groups':len(captures),'conservative_category_scene_groups':len({c['conservative_group'] for c in captures}),'images_by_category':dict(Counter(r['category'] for r in images)),'captures_by_category':dict(Counter(r['category'] for r in captures)),'all_depths_metric_millimeters_verified_from_native_metadata':True,'all_rgb_depth_shapes_match':True,'all_original_files_crc_and_sha256_verified':True,'depth_mask_note':'Native metadata defines finite(depth)>0 with zero invalid. Raw values above10m including65535-like readings are retained unchanged. Evaluation must declare one fixed range before looking at errors.','capture_identity_note':'Numeric category/scene IDs restart across transport shards and visually differ; view_group includes shard. Physical room/location identities are not supplied. Conservative grouping merges repeated category/scene IDs for resampling, but does not prove independence.','lighting_note':'Frame names are ordinal variant IDs, not calibrated illuminant chromaticities. RGBs are processed real camera images; exact camera ISP/white-balance state and physical illuminant RGB vectors are not provided.','exact_duplicate_rgb_groups':duplicate_rgb,'exact_duplicate_depth_capture_groups':duplicate_depth,'captures':captures,'images':images}
 path=OUT/'manifest.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
 print(json.dumps({'manifest':str(path),'sha256':sha(path),'images':len(images),'capture_groups':len(captures),'conservative_groups':manifest['conservative_category_scene_groups'],'images_by_category':manifest['images_by_category'],'exact_duplicate_rgb_groups':len(duplicate_rgb),'exact_duplicate_depth_capture_groups':len(duplicate_depth),'all_native_hashes_verified':True},indent=2),flush=True)

if __name__=='__main__':main()
