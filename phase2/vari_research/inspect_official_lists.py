"""Summarize official path lists without assuming archive contents exist."""
import csv
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
pat=re.compile(r'(?P<scene>\d+)_(?P<camera>c\d+)_(?P<illumination>.+?)(?P<view>\d{4})\.jpg$')
report={}
allgroups={}
for split in ['train','val','test']:
    groups=defaultdict(list);scenes=set();lights=Counter();invalid=[]
    src=ROOT/'dca_reference/data/vari'/f'{split}.txt'
    for row in csv.reader(src.open(encoding='utf8')):
        if not row:continue
        m=pat.search(Path(row[0]).name)
        if not m:invalid.append(row);continue
        d=m.groupdict();key=(d['scene'],d['camera'],d['view'])
        scenes.add(d['scene']);lights[d['illumination']]+=1
        groups[key].append({'rgb':row[0],'normal':row[1],'albedo':row[2],'depth':row[3],**d})
    allgroups[split]=groups
    report[split]={'list_rows':sum(map(len,groups.values())),'distinct_scene_ids':len(scenes),
       'scene_ids':sorted(scenes,key=int),'camera_view_groups':len(groups),
       'illumination_counts':dict(lights),'images_per_camera_view_group':dict(Counter(len(v) for v in groups.values())),
       'unparsed_rows':len(invalid),'sample_group':next(iter(groups.values()))}
trainkeys=set(allgroups['train']);valkeys=set(allgroups['val']);testkeys=set(allgroups['test'])
report['overlap']={'train_val_camera_view_groups':len(trainkeys&valkeys),
    'train_test_scene_ids':len({k[0] for k in trainkeys}&{k[0] for k in testkeys}),
    'train_val_scene_ids':len({k[0] for k in trainkeys}&{k[0] for k in valkeys})}
(ROOT/'official_list_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf8')
compact={k:{a:b for a,b in v.items() if a!='sample_group'} for k,v in report.items()}
print(json.dumps(compact,indent=2))
