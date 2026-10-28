"""Record public asset hashes from an already audited, frozen source snapshot.

This is a provenance-recording command, not part of normal acquisition.
"""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data/vari'
manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
audit=json.loads((DATA/'depth_decode_audit.json').read_text(encoding='utf8'))
groups={r['view_group']:r for r in audit['groups']}
snapshot={'source_repository_commit':'2c8fde439bc94b56783b267198b3fe7e00b23fcf',
    'public_share_url':'https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87',
    'archive_bytes':824263108,'selection_seed':20261028,'images':[]}
for r in manifest['images']:
    g=groups[r['view_group']]
    snapshot['images'].append({k:r[k] for k in ['id','scene','view_group','lighting','split','rgb_sha256','raw_depth_sha256','depth_sha256']}
        | {'legacy_depth_sha256':g['legacy_depth_sha256']})
target=HERE/'selected_asset_hashes.json'
target.write_text(json.dumps(snapshot,indent=2),encoding='utf8')
print(target)
