"""Prefill the student-owned slots of the retained official worksheet PDF."""
from pathlib import Path
import hashlib
import json
import re
import io
import pymupdf as fitz
from PIL import Image

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
OUT = ROOT / 'output/proposal_submission'
WORK = HOME / '.build/official_tables'
STEM = '2023521460130_Shirsha Mitra_Proposal Report Table1_3'
TITLE = 'Design and Implementation of a Color-Constancy-Assisted Monocular Depth Estimation System'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source = next(ROOT.parent.glob('Sichuan University Undergraduate Thesis*Worksheet*.pdf'),
                  ROOT / 'documents/templates/Sichuan_University_Thesis_Worksheet.pdf')
    original_hash = sha(source)
    original = fitz.open(source)
    assert len(original) == 12
    assert 'To be filled in by the supervisor' in original[4].get_text()
    doc = fitz.open()
    doc.insert_pdf(original, from_page=3, to_page=9)
    # Original source is retained; this output intentionally makes its red
    # instructions and logos monochrome, following the student's preference.
    handled = set()
    for page in doc:
        for im in page.get_images(full=True):
            xref = im[0]
            if xref in handled:
                continue
            image = Image.open(io.BytesIO(doc.extract_image(xref)['image'])).convert('L')
            # Replace the shared XObject directly; no temporary page drawing.
            doc.update_object(xref, f'<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} /ColorSpace /DeviceGray /BitsPerComponent 8 >>')
            doc.update_stream(xref, image.tobytes(), compress=True)
            handled.add(xref)
    num = rb'[-+]?(?:\d*\.\d+|\d+\.?\d*)'
    rgb = re.compile(rb'(?<!\S)(' + num + rb')\s+(' + num + rb')\s+(' + num + rb')\s+(rg|RG)(?=\s|$)')
    gray = re.compile(rb'(?<!\S)(' + num + rb')\s+(g|G)(?=\s|$)')
    for page in doc:
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            stream = rgb.sub(lambda m: (b'1 1 1 ' if all(float(m[i]) == 1 for i in (1, 2, 3)) else b'0 0 0 ') + m[4], stream)
            stream = gray.sub(lambda m: (b'1 ' if float(m[1]) == 1 else b'0 ') + m[2], stream)
            doc.update_stream(xref, stream)

    records = []

    def field(page_number, name, rect, value='', owner='student', size=11, multiline=False, checkbox=False):
        w = fitz.Widget()
        w.field_name = name
        w.field_label = ('Supervisor or meeting officials' if owner == 'supervisor' else 'Student') + ': ' + name.replace('_', ' ')
        w.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX if checkbox else fitz.PDF_WIDGET_TYPE_TEXT
        w.rect = fitz.Rect(rect)
        w.field_value = 'Off' if checkbox else value
        w.text_color = (0, 0, 0)
        w.text_font = 'TiRo'
        w.text_fontsize = size
        w.border_width = 0
        if multiline:
            w.field_flags = fitz.PDF_TX_FIELD_IS_MULTILINE
        widget = doc[page_number - 1].add_widget(w)
        doc.xref_set_key(widget.xref, 'F', '4')  # annotation is printable
        records.append({'page': page_number, 'name': name, 'value': 'Off' if checkbox else value,
                        'owner': owner, 'kind': 'checkbox' if checkbox else 'text', 'rect': list(rect)})

    for page_number, offset, prefix in [(1, 0, 't1'), (4, -15.6, 't3')]:
        def box(a, b, c, d):
            return (a, b + offset, c, d + offset)
        field(page_number, prefix + '_title', box(173, 351, 443, 383), TITLE, size=10.5, multiline=True)
        field(page_number, prefix + '_college', box(193, 410, 445, 430), 'College of Software', size=13)
        field(page_number, prefix + '_major', box(184, 457, 449, 477), 'Software Engineering', size=13)
        field(page_number, prefix + '_student_name', box(229, 504, 448, 524), 'Shirsha Mitra', size=13)
        field(page_number, prefix + '_student_id', box(210, 551, 335, 571), '2023521460130', size=12)
        field(page_number, prefix + '_batch', box(377 if page_number == 1 else 371, 551, 440, 571), '2023', size=13)
        field(page_number, prefix + '_supervisor_name', box(210, 598, 441, 618), 'SHAOBING GAO', size=13)
        for key, rect in [('year', (175, 715, 198, 735)), ('month', (252, 715, 266, 735)), ('day', (335, 715, 349, 735))]:
            field(page_number, prefix + '_cover_date_' + key, box(*rect), size=10)

    # Table 2 is explicitly assigned to the supervisor. Every slot stays empty.
    for name, rect, multiline in [
        ('title', (171, 182, 535, 209), True), ('college', (170, 213, 246, 233), False),
        ('major', (317, 213, 381, 233), False), ('batch', (455, 213, 536, 233), False),
        ('purpose_and_tasks', (78, 427, 533, 541), True),
        ('data_and_assigned_readings', (78, 579, 533, 697), True)
    ]:
        field(2, 't2_' + name, rect, owner='supervisor', multiline=multiline)
    for name, rect in [
        ('longitudinal_topic', (285, 252, 294, 262)), ('horizontal_topic', (285, 286, 294, 296)),
        ('teacher_initiated_topic', (293, 326, 302, 336)), ('student_initiated_topic', (293, 374, 302, 384)),
        ('theoretical_research', (442, 276, 451, 286)), ('applied_research', (440, 326, 449, 336)),
        ('technology_development', (439, 383, 448, 393))
    ]:
        field(2, 't2_' + name, rect, owner='supervisor', checkbox=True)
    field(3, 't2_completion_plan', (78, 92, 533, 174), owner='supervisor', multiline=True)
    field(3, 't2_acceptance_date', (221, 187, 326, 207), owner='supervisor', size=10)
    field(3, 't2_requested_completion_date', (455, 187, 535, 207), owner='supervisor', size=10)

    body = {
        'significance': 'Depth can add scene structure to color constancy, but missing or damaged geometry can also mislead a model. I will test whether input-derived depth cues can control how much geometry contributes to illuminant estimation. The immediate module takes XYZRGB points and predicts one global RGB illuminant. Its relationship to the registered monocular depth topic will be confirmed with my supervisor. The completed reduced-budget pilot has not established an accuracy gain. The research question is whether reliability-aware fusion can provide a useful and reproducible benefit under controlled depth degradation.',
        'literature': 'PCCC estimates scene illumination from colored point clouds [1], using a PointNet-based architecture [2]. Earlier 10,000-epoch reproduction runs covered the three released datasets and motivated closer study of depth quality. Learned multimodal gates are already established [3]; adding a gate alone is therefore not sufficient novelty. This study needs comparisons against ordinary fusion, a missing-depth mask, a constant gate and capacity controls. Models that predict geometry from images may later supply additional inputs, provided corresponding images, geometry and illumination references are available.',
        'research_content': '1. Characterize native and degraded depth using the released PCCC captures. The earlier reproduction covered 704 captures across DepthAWB, ETH3D and NYU-v2 plus DIODE.\n\n2. Implement and evaluate input-conditioned geometry fusion. The completed ETH3D pilot trained seven methods over three official folds, giving 21 models at 100 epochs each.\n\n3. Test constant-gate and capacity controls, longer schedules and repeated seeds. Follow with rendered scenes and additional real captures. Preserve negative findings and assess the connection to monocular depth without assuming it is already demonstrated. Experimental outputs are attached.',
        'method': 'Each input contains 256 XYZRGB points. Separate color and geometry branches produce features. A gate uses depth validity, valid-depth fraction and local log-depth disagreement to weight geometry. Color features plus weighted geometry feed a global illuminant head. Labels supervise angular loss and a prediction-consistency term; test errors and corruption identities never enter the gate.\n\nThe pilot used 104 training, 26 inner-validation and 65 test captures per fold, with fixed epoch 100 as the primary checkpoint. All 42 saved checkpoints and 13,650 prediction rows passed audit. Fresh local-GPU inference reproduced 975 predictions exactly. Longer studies need a fixed protocol and appropriate compute. The gate did not establish an improvement over mask fusion or PCCC in this pilot.',
        'schedule': 'Completed by November 2026: source and data audits, full-budget PCCC reproduction, a 21-model reliability-fusion pilot and independent output checks.\n\n25 November 2026: scheduled proposal defense. Record the actual meeting feedback afterwards.\n\nBy 11 December, 11:30: complete and obtain the required signatures on Tables 1-3, then submit the electronic version to the supervisor.\n\nProposed after approval: December 2026-January 2027 for controlled ablations and repeated seeds; February-March for rendered scenes and suitable new real data; April-May for analysis, system integration and thesis writing. Final milestones depend on the supervisor and college schedule.',
        'references': '[1] Xing X et al. Point Cloud Color Constancy. CVPR, 2022, pp. 19750-19759.\n\n[2] Qi C R et al. PointNet Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR, 2017. arXiv:1612.00593.\n\n[3] Arevalo J et al. Gated Multimodal Units for Information Fusion. 2017. arXiv:1702.01992.'
    }
    for name, rect in [('significance', (179, 79, 511, 283)), ('literature', (179, 298, 511, 503)),
                       ('research_content', (179, 519, 511, 743))]:
        field(5, 't3_' + name, rect, body[name], size=11, multiline=True)
    for name, rect in [('method', (161, 80, 517, 283)), ('schedule', (161, 299, 517, 502)),
                       ('references', (161, 518, 517, 736))]:
        field(6, 't3_' + name, rect, body[name], size=11, multiline=True)

    # Table 3 meeting record and supervisor assessment remain entirely blank.
    for name, rect in [('time', (145, 108, 223, 127)), ('place', (316, 108, 399, 127)),
                       ('host', (451, 108, 510, 127))]:
        field(7, 'meeting_' + name, rect, owner='supervisor', size=8, multiline=True)
    for row, y in enumerate([162, 186, 209], 1):
        for name, x0, x1 in [('left_name', 146, 223), ('left_job_title', 226, 312),
                             ('right_name', 317, 400), ('right_job_title', 404, 510)]:
            field(7, f'meeting_teacher_{row}_{name}', (x0, y, x1, y + 20), owner='supervisor', size=9)
    field(7, 'meeting_actual_minutes', (149, 240, 506, 411), owner='supervisor', multiline=True)
    field(7, 'meeting_recorder', (427, 416, 505, 436), owner='supervisor', size=10)
    field(7, 'supervisor_assessment', (149, 459, 506, 644), owner='supervisor', multiline=True)
    for key, rect in [('year', (323, 683, 348, 704)), ('month', (389, 683, 405, 704)), ('day', (454, 683, 471, 704))]:
        field(7, 'supervisor_comment_date_' + key, rect, owner='supervisor', size=10)

    appendix_path = ROOT / 'output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.pdf'
    delivery = json.loads((HOME / 'experiment_report_delivery.json').read_text())
    assert sha(appendix_path) == delivery['outputs_sha256'][appendix_path.name]
    appendix = fitz.open(appendix_path)
    doc.insert_pdf(appendix, from_page=1, to_page=3)
    doc.set_metadata({'title': STEM, 'author': 'Shirsha Mitra',
                      'subject': 'Unsigned prefilled Tables 1-3 for supervisor completion and review',
                      'keywords': 'Official worksheet prefilled student sections unsigned'})
    target = OUT / f'{STEM}.pdf'
    doc.save(target, garbage=4, deflate=True)
    assert sha(source) == original_hash
    receipt = {'status': 'PREFILLED_UNSIGNED_REQUIRES_SUPERVISOR', 'output': str(target), 'output_sha256': sha(target),
               'source': str(source), 'source_sha256': original_hash, 'source_pages_one_based': list(range(4, 11)),
               'form_pages': 7, 'appendix_pages': 3, 'total_pages': len(doc), 'title_unchanged': TITLE,
               'fields': records, 'appendix_sha256': sha(appendix_path),
               'blank_signature_locations': [{'pdf_page': 3, 'roles': ['Student', 'Supervisor', 'College approval']},
                                               {'pdf_page': 7, 'roles': ['Supervisor']}],
               'teacher_owned_fields_filled': 0, 'signatures_added': 0}
    (HOME / 'official_tables_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': receipt['status'], 'pages': len(doc), 'fillable_fields': len(records),
                      'teacher_owned_fields_filled': 0, 'signatures_added': 0, 'output': str(target)}, indent=2))
    for i, page in enumerate(fitz.open(target), 1):
        page.get_pixmap(dpi=110, colorspace=fitz.csRGB, annots=True).save(WORK / f'page-{i:02}.png')


if __name__ == '__main__':
    main()
