"""Final content/data checks and a private, hash-verified defense bundle."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import subprocess
import sys
from xml.etree import ElementTree as ET
import fitz

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
OUT=ROOT/'output/defense_2026_11_25'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (OUT/'Shirsha_Mitra_Proposal_Report_With_Experiments.pdf').exists():
        subprocess.run([sys.executable, str(HOME/'verify_experiment_report.py')], check=True, cwd=ROOT)
        return
    d=json.loads((HOME/'content.json').read_text())
    ppt=OUT/'Shirsha_Mitra_Proposal_Defense_2026.pptx'
    receipt=json.loads((HOME/'.build/validation-final4.json').read_text())
    assert sha(ppt)==receipt['finalSha256']
    assert receipt['packageIntegrity']['status']=='pass' and receipt['firstPartyImport']['passed']
    assert receipt['nativeChartValidation']['passed']
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with zipfile.ZipFile(ppt) as z:
        slides=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n))
        notes=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n))
        assert len(slides)==len(notes)==7
        texts=[' '.join(t.text or '' for t in ET.fromstring(z.read(n)).findall('.//a:t',ns)) for n in slides]
        assert 'MLP(depth cues)' in texts[2]
        assert 'Both primary intervals include zero' in texts[3]
        assert all('meeting.tencent.com/' not in t for t in texts)
        charts=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/charts/chart\d+.xml',n))
        expected=[[[r['real'] for r in d['historical']],[r['flat'] for r in d['historical']]],
                  [[d['metrics'][m]['clean'] for m in ['pccc','mask','learned']],
                   [d['metrics'][m]['damaged_mean'] for m in ['pccc','mask','learned']]]]
        assert len(charts)==2
        for name,series_expected in zip(charts,expected):
            series=ET.fromstring(z.read(name)).findall('.//c:ser',ns)
            for item,values in zip(series,series_expected):
                actual=[float(v.text) for v in item.findall('.//c:val//c:numCache/c:pt/c:v',ns)]
                assert len(actual)==len(values)==3
                assert all(abs(a-b)<1e-8 for a,b in zip(actual,values))
        table=ET.fromstring(z.read('ppt/slides/slide6.xml'))
        cells=[t.text or '' for t in table.findall('.//a:tbl//a:t',ns)]
        for m in d['order']:
            for c in ['clean','noise10','noise30','drop50','shuffle']:assert f"{d['metrics'][m][c]:.3f}" in cells
    with zipfile.ZipFile(OUT/'Shirsha_Mitra_Proposal_Report.docx') as z:
        nsword={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        document=ET.fromstring(z.read('word/document.xml'))
        text=' '.join(x.text or '' for x in document.findall('.//w:t',nsword))
        for heading in ['Significance of the Topic','Literature Review','Main Research Content',
                        'Proposed Methodology and Technical Roadmap','Work Plan and Schedule','References']:
            assert heading in text
        assert not ET.fromstring(z.read('word/styles.xml')).findall('.//w:pBdr',nsword)
    pdf=fitz.open(OUT/'Shirsha_Mitra_Proposal_Report.pdf')
    assert len(pdf)==6
    text=' '.join(p.get_text() for p in pdf)
    assert all(v in text for v in ['13,650','2.574','2.588','2.318','26','65','23 November 2026'])
    assert '\ufffd' not in text and '\u25a0' not in text
    assert sum(s['seconds'] for s in d['script'])==180 and d['talk_word_count']==249
    files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in ('.pptx','.docx','.pdf','.txt')]
    private=OUT/'private/Meeting_Details.txt'
    assert private.exists()
    file_hashes={p.name:sha(p) for p in files}
    file_hashes['private/Meeting_Details.txt']=sha(private)
    manifest=dict(prepared_date='2026-11-23',defense_date='2026-11-25',main_slides=5,backup_slides=2,
                  target_speaking_seconds=180,script_words=249,proposal_pages=6,
                  native_powerpoint_open_and_render_verified=True,native_word_render_verified=True,
                  native_chart_edit_save_reopen_test=True,
                  table_layout_warning='Estimator warning reviewed against native PowerPoint renders; no overlap or clipping',
                  research_status='Audited 100-epoch development pilot; no established gating benefit; full research remains ongoing',
                  output_sha256=file_hashes,evidence_sha256=d['evidence_sha256'],
                  verifier_sha256=sha(Path(__file__)))
    (HOME/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    archive=OUT/'Defense_Pack_Private.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.write(private,'private/Meeting_Details.txt')
        z.writestr('File_Hashes.json',json.dumps(file_hashes,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in file_hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    print(json.dumps(dict(status='PASS',private_bundle=str(archive),files=len(file_hashes),
                         slide_count=7,main_slides=5,proposal_pages=6,script_words=249),indent=2))


if __name__=='__main__':main()
