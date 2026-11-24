"""Verify current illustrated report and update the private defense bundle."""
from pathlib import Path
import hashlib
import io
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
import pymupdf
import numpy as np
from PIL import Image

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
OUT = ROOT / 'output/defense_2026_11_25'
STEM = 'Shirsha_Mitra_Proposal_Report_With_Experiments'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
METHODS = ['rgb', 'ordinary', 'mask', 'learned', 'learned_clean', 'learned_no_consistency', 'pccc']
CONDITIONS = ['clean', 'noise10', 'noise30', 'drop50', 'shuffle']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def normalize(text):
    return ' '.join(text.replace('\u00ad', '').split())


def main():
    evidence = read(HOME / 'experiment_report_evidence.json')
    build = read(HOME / 'experiment_report_build.json')
    native = read(HOME / 'experiment_report_native_render.json')
    pdf_receipt = read(HOME / 'experiment_report_pdf_export.json')
    docx, pdf = [OUT / f'{STEM}.{ext}' for ext in ('docx', 'pdf')]
    assert evidence['collector_sha256'] == sha(HOME / 'prepare_report_evidence.py')
    assert build['docx_sha256'] == native['source_sha256'] == pdf_receipt['source_sha256'] == sha(docx)
    assert pdf_receipt['pdf_sha256'] == sha(pdf)
    assert build['evidence_sha256'] == sha(HOME / 'experiment_report_evidence.json')
    assert native['status'] == 'PASS' and native['pages'] == 4
    for name, digest in evidence['evidence_sha256'].items():
        assert sha(ROOT / 'phase9_reliability_fusion/pilot_v1' / name) == digest
    assert len(evidence['completions']) == 21
    checkpoint_count = 0
    for job in evidence['completions']:
        folder = ROOT / f"phase9_reliability_fusion/runs/eth3d_pilot_v1/{job['method']}/fold{job['fold']}"
        assert sha(folder / 'history.json') == job['history_sha256']
        for name, digest in job['checkpoint_sha256'].items():
            assert sha(folder / name) == digest
            checkpoint_count += 1
    assert checkpoint_count == 42
    assert evidence['fresh_inference']['predictions_reproduced_exactly'] == 975
    assert evidence['fresh_inference']['maximum_component_discrepancy'] == 0
    assert not evidence['fresh_inference']['modal_used'] and evidence['fresh_inference']['retrained_models'] == 0
    with zipfile.ZipFile(docx) as z:
        assert z.testzip() is None
        document = ET.fromstring(z.read('word/document.xml'))
        body = document.find('w:body', NS)
        docx_text = normalize(' '.join(n.text or '' for n in body.findall('.//w:t', NS)))
        tables = document.findall('.//w:tbl', NS)
        assert len(tables) == 6
        for table in tables:
            for border in table.findall('.//w:tcBorders/*', NS):
                assert border.get(f'{{{W}}}color') == '000000'
            for shade in table.findall('.//w:shd', NS):
                assert shade.get(f'{{{W}}}fill') == 'FFFFFF'
        for name in ('word/document.xml', 'word/styles.xml', 'word/footer1.xml'):
            node = ET.fromstring(z.read(name))
            assert not node.findall('.//w:pBdr', NS)
            assert all(c.get(f'{{{W}}}val') == '000000' for c in node.findall('.//w:color', NS))
        media = [n for n in z.namelist() if n.startswith('word/media/')]
        assert len(media) == 4
        for name in media:
            assert Image.open(io.BytesIO(z.read(name))).mode == 'L'
        all_rows = tables[2].findall('w:tr', NS)
        for row, method in zip(all_rows[1:], METHODS):
            cells = [normalize(' '.join(t.text or '' for t in c.findall('.//w:t', NS))) for c in row.findall('w:tc', NS)]
            expected = [f"{next(r['mean'] for r in evidence['summary'] if r['method'] == method and r['condition'] == c and r['selection'] == 'fixed_final'):.3f}" for c in CONDITIONS]
            assert cells[1:] == expected
        example_rows = tables[4].findall('w:tr', NS)[1:]
        expected_examples = [(e, condition) for e in evidence['examples'] for condition in ('clean', 'drop50')]
        for row, (example, condition) in zip(example_rows, expected_examples):
            cells = [normalize(' '.join(t.text or '' for t in c.findall('.//w:t', NS))) for c in row.findall('w:tc', NS)]
            assert cells[0] == example['id']
            assert cells[2:] == [f"{example['predictions'][m][condition]['error_deg']:.3f}" for m in ('learned', 'pccc')]
    pdf_doc = pymupdf.open(pdf)
    assert len(pdf_doc) == 4
    pdf_text = ' '.join(re.sub(r'Shirsha Mitra\s+[1-4]', '', p.get_text()) for p in pdf_doc)
    assert normalize(pdf_text) == docx_text, 'PDF paragraphs and all table cells must match DOCX text'
    for value in ('Shirsha Mitra', '2023521460130', 'SHAOBING GAO', 'College of Software', 'Software Engineering', '2023', '24 November 2026', '13,650', '975'):
        assert value in docx_text and value in normalize(pdf_text)
    assert 'Both intervals include zero' in docx_text and 'does not establish a gating benefit' in docx_text
    assert not any(c in pdf_text for c in ('\ufffd', '\u25a0'))
    image_counts = []
    for page in pdf_doc:
        image_counts.append(len(page.get_images()))
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    assert span['color'] == 0
                    box = pymupdf.Rect(span['bbox'])
                    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= page.rect.width + 1 and box.y1 <= page.rect.height + 1
        assert all(im[5] == 'DeviceGray' for im in page.get_images(full=True))
    assert image_counts == [0, 2, 1, 1]
    visual = read(HOME / '.build/experiment_report_visual_review.json')
    assert visual['status'] == 'PASS' and visual['docx_sha256'] == sha(docx) and visual['pdf_sha256'] == sha(pdf)
    assert visual['word_pages_reviewed'] == visual['pdf_pages_reviewed'] == [1, 2, 3, 4]
    pptx = OUT / 'Shirsha_Mitra_Proposal_Defense_2026.pptx'
    assert sha(pptx) == read(HOME / '.build/validation-final4.json')['finalSha256']
    files = [pptx, docx, pdf] + [OUT / n for n in ('Three_Minute_Script_English_and_Bengali.txt', 'Defense_QA_English_and_Bengali.txt', 'START_HERE.txt', 'QQ_Draft_English.txt', 'private/Meeting_Details.txt')]
    archive = OUT / 'Defense_Pack_Private.zip'
    backup = HOME / '.build/Defense_Pack_Private_2026_11_23.zip'
    if archive.exists() and not backup.exists():
        shutil.copy2(archive, backup)
    file_hashes = {p.relative_to(OUT).as_posix(): sha(p) for p in files}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(OUT).as_posix())
        z.writestr('File_Hashes.json', json.dumps(file_hashes, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in file_hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        assert not any(n.endswith('Proposal_Report.docx') or n.endswith('Proposal_Report.pdf') for n in z.namelist())
    result = {'status': 'PASS', 'updated_date': '2026-11-24', 'report_pages': 4, 'experimental_figures': 4,
              'tables': 6, 'text_and_borders_black': True, 'figures_grayscale': True, 'pdf_text_matches_docx': True,
              'completed_models': 21, 'checkpoint_hashes_rechecked': 42, 'fresh_local_gpu_predictions_reproduced': 975,
              'new_training': False, 'modal_used_for_this_update': False,
              'source_evidence_sha256': sha(HOME / 'experiment_report_evidence.json'),
              'outputs_sha256': file_hashes, 'archive_sha256': sha(archive), 'verifier_sha256': sha(Path(__file__))}
    (HOME / 'experiment_report_delivery.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    main()
