"""Read-only audit of the frozen exploratory Vari manifest and selected assets.

Writes its report only under vari_research. It never rewrites dataset files.
"""
import binascii
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
DATA=HERE.parent/'data/vari'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)

def main():
    manifest_path=DATA/'manifest.json';initial_manifest_sha=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'));images=manifest['images']
    expected_snapshot=json.loads((HERE/'selected_asset_hashes.json').read_text(encoding='utf8'))
    expected_images={r['id']:r for r in expected_snapshot['images']}
    require({r['id'] for r in images}==set(expected_images),'SelectedimageIDs differfromrecordedpublicsourcesnapshot')
    for r in images:
        expected=expected_images[r['id']]
        for key in ['scene','view_group','lighting','split','rgb_sha256','raw_depth_sha256','depth_sha256']:
            require(r[key]==expected[key],'Manifest differsfromrecordedsourcesnapshot:'+r['id']+':'+key)
    depth_audit=json.loads((DATA/'depth_decode_audit.json').read_text(encoding='utf8'))
    audit_by_group={g['view_group']:g for g in depth_audit['groups']}
    central=json.loads((DATA/'zip_central_directory.json').read_text(encoding='utf8'))
    members={r['name']:r for r in central}
    declared=json.loads((DATA/'exploratory_split_groups.json').read_text(encoding='utf8'))
    lights={'Environment','Indoor','SunNoon_Indoor','SunNight_Indoor','SunMorning_Indoor',
       'SunNoon_Indoor_Environment','SunNight_Indoor_Environment','SunMorning_Indoor_Environment',
       'SunNoon','SunNight','SunMorning'}
    groups=defaultdict(list)
    for r in images:groups[r['view_group']].append(r)
    require(len(images)==528,'Expected528images')
    require(len(groups)==48,'Expected48groups')
    require(len({r['id'] for r in images})==528,'ImageIDsnotunique')
    counts=Counter(r['split'] for r in images)
    require(dict(counts)=={'training':220,'validation':44,'evaluation':264},'Incorrectsplitimagecounts')
    group_counts=Counter(v[0]['split'] for v in groups.values())
    require(dict(group_counts)=={'training':20,'validation':4,'evaluation':24},'Incorrectgroupcounts')
    expected_scenes={'training':{'212'},'validation':{'212'},'evaluation':{'485'}}
    scene_sets={s:{r['scene'] for r in images if r['split']==s} for s in expected_scenes}
    require(scene_sets==expected_scenes,'Incorrectroomassignment')
    group_sets={s:{r['view_group'] for r in images if r['split']==s} for s in expected_scenes}
    require(not(group_sets['training']&group_sets['validation']),'Trainingvalidationviewoverlap')
    require(not((group_sets['training']|group_sets['validation'])&group_sets['evaluation']),'Evaluationviewoverlap')
    checked_raw=set();checked_arrays=set();mean_differences=[];maximum_difference=0;raw_ranges=[]
    def verify_raw(p,expected_sha):
        p=p.resolve();require(p.is_relative_to((DATA/'raw').resolve()),'Rawpathescapesdatafolder')
        require(p.exists(),'Missingrawfile:'+str(p));b=p.read_bytes()
        require(hashlib.sha256(b).hexdigest()==expected_sha,'SHA256mismatch:'+str(p))
        member=p.relative_to(DATA/'raw').as_posix();entry=members[member]
        require(len(b)==entry['bytes'],'ZIPmemberlengthmismatch:'+member)
        require(f'{binascii.crc32(b)&0xffffffff:08x}'==entry['crc32'],'ZIPCRC32mismatch:'+member)
        checked_raw.add(member)
    for gid,rows in groups.items():
        require(len(rows)==11 and {r['lighting'] for r in rows}==lights,'Incompleteilluminationgroup:'+gid)
        require(len({r['split'] for r in rows})==1,'Viewgroupsplitacrosspartitions:'+gid)
        for key in ['scene','depth','legacy_depth','depth_sha256','raw_depth_sha256']:
            require(len({r[key] for r in rows})==1,'Illuminationreferenceinconsistency:'+gid+':'+key)
        g=audit_by_group[gid];raw=PROJECT/g['raw_depth'];verify_raw(raw,g['raw_sha256'])
        original=tifffile.imread(raw)
        require(original.dtype==np.uint16,'RawTIFFisnotuint16:'+gid)
        require(original.shape==(480,640,3),'UnexpectedrawTIFFshape:'+gid)
        require(np.array_equal(original[...,0],original[...,1]) and np.array_equal(original[...,0],original[...,2]),'Nonidenticaldepthchannels:'+gid)
        primary=PROJECT/rows[0]['depth'];legacy=PROJECT/rows[0]['legacy_depth']
        require(sha(primary)==g['depth_sha256']==rows[0]['depth_sha256'],'Primarydepthhashmismatch:'+gid)
        require(sha(legacy)==g['legacy_depth_sha256'],'Legacydepthhashmismatch:'+gid)
        require(g['legacy_depth_sha256']==expected_images[rows[0]['id']]['legacy_depth_sha256'],'Legacyhashdiffersfromrecordedsource:'+gid)
        pred16=np.load(primary,allow_pickle=False);pred8=np.load(legacy,allow_pickle=False)
        expected16=((1-original[...,0].astype(np.float32)/65535)*10).astype(np.float32)
        with Image.open(raw) as im:expected8=((1-np.asarray(im.convert('L'),np.float32)/255)*10).astype(np.float32)
        require(np.array_equal(pred16,expected16),'Primary16bitdecodemismatch:'+gid)
        require(np.array_equal(pred8,expected8),'AuthorsPIL8decodemismatch:'+gid)
        require(pred16.shape==(480,640) and pred16.dtype==np.float32 and np.isfinite(pred16).all(),'Primaryarrayinvalid:'+gid)
        diff=np.abs(pred16-pred8);mean_differences.append(float(diff.mean()));maximum_difference=max(maximum_difference,float(diff.max()))
        raw_ranges.append([int(original.min()),int(original.max())]);checked_arrays.update([str(primary),str(legacy)])
        for row in rows:
            rgb=PROJECT/row['rgb'];verify_raw(rgb,row['rgb_sha256'])
            with Image.open(rgb) as im:require(im.size==(640,480) and im.mode=='RGB','RGBsizeormodemismatch:'+row['id'])
    # Reproduce the declared seeded split using only the frozen complete index.
    complete=json.loads((DATA/'complete_groups.json').read_text())
    rng=np.random.default_rng(20261028)
    room212=sorted([g for g in complete['groups'] if g['scene']=='212'],key=lambda g:g['view_group'])
    room485=sorted([g for g in complete['groups'] if g['scene']=='485'],key=lambda g:g['view_group'])
    shuffled212=[room212[i] for i in rng.permutation(len(room212))]
    shuffled485=[room485[i] for i in rng.permutation(len(room485))]
    expected_groups={'training':{g['view_group'] for g in shuffled212[:20]},
        'validation':{g['view_group'] for g in shuffled212[20:24]},'evaluation':{g['view_group'] for g in shuffled485[:24]}}
    require(group_sets==expected_groups,'Seededsplitnotreproducible')
    require(sha(manifest_path)==initial_manifest_sha,'Manifestchangedduringaudit')
    report={'status':'pass','audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'read_only':True,'manifest_sha256':initial_manifest_sha,'images':len(images),'view_groups':len(groups),
        'groups_by_split':dict(group_counts),'images_by_split':dict(counts),
        'rooms_by_split':{k:sorted(v) for k,v in scene_sets.items()},
        'no_training_evaluation_room_overlap':True,'no_view_group_overlap_between_any_split':True,
        'all_groups_have_exactly11_lights':True,'raw_members_SHA256_and_ZIP_CRC32_verified':len(checked_raw),
        'decoded_NPY_arrays_SHA256_and_exact_formula_verified':len(checked_arrays),
        'all_within_group_reference_paths_and_hashes_identical':True,'seeded_split_reproduced':True,
        'recorded_public_asset_snapshot_matched':True,
        'raw_depth_dtype':'uint16','raw_depth_shape':[480,640,3],
        'raw_depth_channels_bitidentical':True,'raw_depth_value_range_over_selected_groups':[min(r[0] for r in raw_ranges),max(r[1] for r in raw_ranges)],
        'primary_decoder':'(1-uint16/65535)*10m','legacy_decoder':'PILconvertL8bit,then(1-gray/255)*10m',
        'primary_legacy_mean_absolute_difference_m':float(np.mean(mean_differences)),
        'primary_legacy_max_absolute_difference_m':maximum_difference,
        'complete_official_test_groups':complete['complete_groups'],'source_rooms':complete['groups_per_scene'],
        'full_zip_present':(DATA/'test_dataset.zip').exists(),
        'limitation':'Theselectedsubsetiscomplete. FullZIPassemblyisoptionalfortrainingthissubsetandwasnotclaimedcomplete.'}
    target=HERE/'frozen_vari_integrity_audit.json';target.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
