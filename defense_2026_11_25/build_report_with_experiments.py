"""Build the concise monochrome proposal from verified experiment artifacts."""
from pathlib import Path
import hashlib
import json
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
OUT = ROOT / 'output/defense_2026_11_25'
FIG = HOME / '.build/experiment_report_figures'
STEM = 'Shirsha_Mitra_Proposal_Report_With_Experiments'
TITLE = 'Design and Implementation of a Color-Constancy-Assisted Monocular Depth Estimation System'
FOCUS = 'Depth Reliability Aware Fusion for Point Cloud Color Constancy'
LABELS = {'rgb': 'RGB only', 'ordinary': 'Ordinary fusion', 'mask': 'Mask fusion',
          'learned': 'Learned gate', 'learned_clean': 'Gate without damage training',
          'learned_no_consistency': 'Gate without consistency', 'pccc': 'Original PCCC'}
CONDITIONS = ['clean', 'noise10', 'noise30', 'drop50', 'shuffle']
BLACK = RGBColor(0, 0, 0)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data = json.loads((HOME / 'experiment_report_evidence.json').read_text())
    assert data['fresh_inference']['predictions_reproduced_exactly'] == 975
    assert data['audit']['status'] == data['independent_verification']['status'] == 'PASS'
    assert data['collector_sha256'] == sha(HOME / 'prepare_report_evidence.py')
    for name, digest in data['evidence_sha256'].items():
        assert sha(ROOT / 'phase9_reliability_fusion/pilot_v1' / name) == digest
    figures = json.loads((FIG / 'manifest.json').read_text())
    for record in figures.values():
        assert sha(Path(record['path'])) == record['sha256']

    def value(method, condition):
        return next(r['mean'] for r in data['summary'] if r['method'] == method and r['selection'] == 'fixed_final' and r['condition'] == condition)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = Cm(1.65)
    sec.bottom_margin = Cm(1.6)
    sec.header_distance = sec.footer_distance = Cm(.65)
    for style in doc.styles:
        if hasattr(style, 'font'):
            style.font.name = 'Times New Roman'
            style.font.color.rgb = BLACK
            rpr = style.element.get_or_add_rPr()
            if rpr.rFonts is not None:
                for attr in list(rpr.rFonts.attrib):
                    if 'theme' in attr.lower():
                        rpr.rFonts.attrib.pop(attr, None)
                for field in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
                    rpr.rFonts.set(qn('w:' + field), 'Times New Roman')
            for color in rpr.findall(qn('w:color')):
                for attr in ('themeColor', 'themeTint', 'themeShade'):
                    color.attrib.pop(qn('w:' + attr), None)
    normal = doc.styles['Normal']
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.05
    normal.paragraph_format.space_after = Pt(5)
    doc.styles['Title'].font.size = Pt(18)
    doc.styles['Title'].font.bold = True
    doc.styles['Title'].paragraph_format.space_after = Pt(5)
    doc.styles['Title'].paragraph_format.space_before = Pt(0)
    for name, size in [('Heading 1', 13), ('Heading 2', 11.5)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(8)
        doc.styles[name].paragraph_format.space_after = Pt(4)
        doc.styles[name].paragraph_format.keep_with_next = True
    for border in list(doc.styles.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
    doc.core_properties.author = 'Shirsha Mitra'
    doc.core_properties.title = 'Undergraduate Thesis Proposal with Experimental Results'
    doc.core_properties.subject = FOCUS
    doc.core_properties.keywords = 'PCCC ETH3D depth reliability fusion proposal'

    def p(text, bold=False, size=None, after=None):
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.bold = bold
        run.font.color.rgb = BLACK
        if size:
            run.font.size = Pt(size)
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        return para

    def h(text, level=1):
        return doc.add_heading(text, level)

    def table(headers, rows, widths, bold_rows=()):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for col, width in zip(t.columns, widths):
            col.width = Cm(width)
        for i, texts in enumerate([headers] + rows):
            row = t.rows[0] if i == 0 else t.add_row()
            for j, (cell, text, width) in enumerate(zip(row.cells, texts, widths)):
                cell.width = Cm(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text = str(text)
                tcpr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for edge in ('top', 'left', 'bottom', 'right'):
                    border = OxmlElement('w:' + edge)
                    for attr, val in [('val', 'single'), ('sz', '4'), ('color', '000000')]:
                        border.set(qn('w:' + attr), val)
                    borders.append(border)
                tcpr.append(borders)
                shade = OxmlElement('w:shd')
                shade.set(qn('w:val'), 'clear')
                shade.set(qn('w:fill'), 'FFFFFF')
                tcpr.append(shade)
                margins = OxmlElement('w:tcMar')
                for edge, padding in [('top', '65'), ('bottom', '65'), ('left', '85'), ('right', '85')]:
                    el = OxmlElement('w:' + edge)
                    el.set(qn('w:w'), padding)
                    el.set(qn('w:type'), 'dxa')
                    margins.append(el)
                tcpr.append(margins)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.space_before = Pt(1)
                    para.paragraph_format.line_spacing = 1
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.size = Pt(10)
                        run.font.color.rgb = BLACK
                        run.bold = i == 0 or (i - 1) in bold_rows
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t

    def picture(name, caption, alt):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.keep_with_next = True
        image = para.add_run().add_picture(str(FIG / f'{name}.png'), width=Cm(17.1))
        image._inline.docPr.set('descr', alt)
        cap = p(caption, size=9.5, after=5)
        cap.paragraph_format.line_spacing = 1

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('Shirsha Mitra    ')
    run.font.size = Pt(9)
    run.font.color.rgb = BLACK
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)

    # Page 1: concise proposal with the registered title preserved.
    doc.add_paragraph('Undergraduate Thesis Proposal Report', style='Title')
    p(FOCUS, bold=True, size=12)
    p('Registered thesis title  ' + TITLE, size=10.5)
    p('Student  Shirsha Mitra     Student ID  2023521460130\n'
      'Supervisor  SHAOBING GAO     Batch  2023\n'
      'Sichuan University     College of Software\n'
      'Department and major  Software Engineering\n'
      'Updated  24 November 2026     Proposal defense  25 November 2026', size=10.5)
    h('Significance of the Topic')
    p('Depth can help color constancy by adding scene structure. Missing or damaged depth can also mislead the model. I am testing whether a model can reduce its use of geometry when the depth input is unreliable. The completed pilot supports further investigation, but does not establish an accuracy improvement.')
    h('Literature Review')
    p('PCCC estimates illumination from colored 3D points [1], using a PointNet based architecture [2]. Learned multimodal gates are already established [3]. The research contribution must therefore come from a useful depth reliability mechanism and convincing controls, rather than the presence of a gate alone.')
    h('Main Research Content')
    p('The study compares RGB only, ordinary fusion, validity mask fusion and learned geometry gating. Ablations test damage training and prediction consistency. Later experiments will add a constant gate, longer training, repeated seeds, rendered scenes and new real captures.')
    h('Proposed Methodology and Technical Roadmap')
    p('Each input has 256 XYZRGB points. Separate color and geometry branches produce point features. A gate uses depth validity, valid point fraction and local log depth disagreement to weight geometry before predicting one global RGB illuminant. Labels supervise the loss; test errors and damage labels never enter the gate.')
    p('Fusion rule  Fused feature = Color feature + gate x Geometry feature', bold=True, size=10.5)
    p('The candidate has 34,020 active parameters; the original PCCC has about 1.68 million. The gate is a mixing weight, not a calibrated confidence score. The current module estimates illumination from RGB-D; its connection to the registered monocular depth topic still needs supervisor agreement.', size=10.5)
    h('Earlier PCCC Reproduction', level=2)
    table(['Dataset', 'Captures', 'Native XYZ', 'Ray flat XYZ'],
          [[r['dataset'], int(r['captures']), f"{r['real']:.2f}", f"{r['flat']:.2f}"] for r in data['historical']],
          [6.5, 2.3, 4.3, 4.3])
    p('Mean angular error in degrees. These are earlier 10,000-epoch Modal T4 runs with the released head and fixed final checkpoints. Ray flat retains camera ray layout. They are separate from the new 100-epoch pilot.', size=9.5)

    # Page 2: actual released input and measured training histories.
    doc.add_page_break()
    h('Experimental Setup and Training')
    p('The new pilot is complete: seven methods were trained on the local Intel Arc 140T GPU across all three official ETH3D folds. Each capture appears in one outer test fold. The table and figures below describe this pilot only.')
    table(['Item', 'Setting'], [
        ['Data', '195 ETH3D captures, 256 XYZRGB points each'],
        ['Per fold split', '104 training, 26 inner validation, 65 test captures'],
        ['Training', '100 epochs, batch 16, Adam, learning rate 0.0003'],
        ['Runs and checkpoint', '7 methods x 3 folds; one seed per fold; fixed epoch 100'],
        ['Test conditions', 'Clean, log noise 0.10 and 0.30, half depth removed, XYZ shuffled'],
        ['Completed audit', '42 checkpoints and 13,650 prediction rows verified']
    ], [4.0, 13.4])
    picture('input_clouds',
            f"Figure 1  Actual released input {data['examples'][0]['id']} and two frozen test interventions. Grayscale is derived from RGB. Invalid depth points are omitted. These are sparse point cloud views, not full photographs.",
            'Three views of the same real 256 point input: clean geometry, lognormal depth noise and removal of half the valid depth.')
    picture('training_curves',
            'Figure 2  Measured training objective and inner validation error, averaged across the three folds. Curves use all saved epochs without smoothing. Test errors were not used for checkpoint selection.',
            'Unsmoothed training and inner validation curves for PCCC, mask fusion and the learned gate from epochs one to one hundred.')
    p('Damage is applied to XYZ while RGB stays unchanged. These are synthetic interventions on real captures. A rendered scene study has not yet been completed.', size=10)

    # Page 3: all outcomes, including the two predeclared uncertainty intervals.
    doc.add_page_break()
    h('Pilot Results')
    p('Table 1 reports pooled mean recovery angular error over 195 test captures. All methods use the same pilot splits, schedule and backend. Lower values are better. PCCC is an architectural baseline with different capacity; the mask and ordinary fusion controls isolate changes within the new design.')
    table(['Method', 'Clean', 'Noise\n0.10', 'Noise\n0.30', 'Half depth\nremoved', 'XYZ\nshuffled'],
          [[LABELS[m]] + [f'{value(m, c):.3f}' for c in CONDITIONS] for m in LABELS],
          [6.0, 2.05, 2.05, 2.05, 2.6, 2.65], bold_rows=(6,))
    p('Table 1  Fixed final epoch 100. Noise uses positive multiplicative depth factors with log space standard deviation 0.10 or 0.30. Half removal affects half of the valid geometry points.', size=9.5)
    picture('noise_results',
            'Figure 3  Sensitivity to increasing depth noise. RGB only is unchanged because its input does not use XYZ. PCCC has the lowest mean error at all three plotted noise levels.',
            'Black line plot of clean, sigma point one and sigma point three depth noise errors for RGB only, PCCC, mask fusion and learned gating.')
    h('Does the Learned Gate Help', level=2)
    pair_rows = []
    for pair in data['paired']:
        endpoint = 'Clean input' if pair['endpoint'] == 'clean' else 'Average of four damaged inputs'
        lo, hi = pair['conditional_capture_bootstrap95']
        pair_rows.append([endpoint, f"{pair['mean_deg']:+.4f}", f'[{lo:+.4f}, {hi:+.4f}]'])
    table(['Endpoint', 'Gate minus mask', 'Conditional 95% interval'], pair_rows, [7.3, 4.1, 6.0])
    p('Negative differences favor the gate. Both intervals include zero. They measure capture resampling conditional on these models, not uncertainty across seeds or rooms; no multiplicity adjustment was applied.', size=10)
    p(f"The learned gate gives {value('learned', 'clean'):.3f} degrees on clean inputs, compared with {value('mask', 'clean'):.3f} for mask fusion and {value('pccc', 'clean'):.3f} for PCCC. The pilot does not establish a gating benefit, and the gate does not beat PCCC. Longer training alone cannot be assumed to change that finding.")

    # Page 4: concrete output vectors, integrity evidence and a compact research plan.
    doc.add_page_break()
    h('Model Outputs and Research Plan')
    picture('illuminant_outputs',
            'Figure 4  Actual clean input illuminant outputs against the reference label. Components are divided by R + G + B. The examples are the first and last fold 0 test IDs in alphabetical order, chosen without using prediction error.',
            'Two examples of reference, learned gate and PCCC illuminant vectors, shown with monochrome grouped bars for R, G and B.')
    example_rows = []
    for example in data['examples']:
        for condition, label in [('clean', 'Clean'), ('drop50', 'Half depth removed')]:
            predicted = example['predictions']
            example_rows.append([example['id'], label, f"{predicted['learned'][condition]['error_deg']:.3f}",
                                 f"{predicted['pccc'][condition]['error_deg']:.3f}"])
    table(['Capture', 'Input', 'Learned gate', 'PCCC'], example_rows, [4.0, 6.0, 3.7, 3.7])
    p('Table 2  Angular error in degrees for the two illustrated captures. Individual examples show model behavior; the pooled results in Table 1 determine the comparison.', size=9.5)
    h('Verification and Limits', level=2)
    p('The audit verified all 42 saved checkpoints and independently recomputed 13,650 prediction errors. A fresh local GPU run reproduced 975 predictions exactly from three fold 0 models across all five test conditions. No new training or Modal run was needed for these illustrations.', size=10.5)
    p('The pilot uses 100 epochs and one seed per fold on previously inspected captures. Capture folds do not prove room independence. PCCC also showed a strict CPU/XPU gradient mismatch; repeatability on the same XPU passed, but cross backend gradient equivalence is not established.', size=10.5)
    h('Feasibility and Work Plan', level=2)
    table(['Stage', 'Next work'], [
        ['Before the proposal defense', 'Confirm scope and rehearse the 3 minute presentation'],
        ['Next controlled study', 'Add a constant gate and capacity controls; predefine longer schedules and repeated seeds'],
        ['Generalization and thesis', 'Test rendered scenes and new real captures; assess image predicted geometry if correspondence and labels are available']
    ], [5.0, 12.4])
    h('References', level=2)
    for text in [
        '[1] Xing X et al. Point Cloud Color Constancy. CVPR, 2022, 19750-19759.',
        '[2] Qi C R et al. PointNet Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR, 2017. arXiv:1612.00593.',
        '[3] Arevalo J et al. Gated Multimodal Units for Information Fusion. 2017. arXiv:1702.01992.'
    ]:
        p(text, size=9.5, after=3)

    # Explicit black formatting removes any inherited theme/accent colors.
    for part in [doc.element, doc.styles.element, sec.footer._element]:
        for element in part.xpath('.//w:color'):
            element.set(qn('w:val'), '000000')
            for attr in ('themeColor', 'themeTint', 'themeShade'):
                element.attrib.pop(qn('w:' + attr), None)
        for element in part.xpath('.//w:shd'):
            element.set(qn('w:fill'), 'FFFFFF')
        for border in list(part.xpath('.//w:pBdr')):
            border.getparent().remove(border)
    path = OUT / f'{STEM}.docx'
    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    words = len(re.findall(r"\b[\w'-]+\b", '\n'.join(p.text for p in doc.paragraphs)))
    receipt = {'status': 'BUILT_AWAITING_RENDER', 'document': str(path), 'planned_pages': 4,
               'figures': len(figures), 'tables': len(doc.tables), 'prose_word_count': words,
               'docx_sha256': sha(path), 'evidence_sha256': sha(HOME / 'experiment_report_evidence.json'),
               'figure_sha256': {k: r['sha256'] for k, r in figures.items()},
               'student': 'Shirsha Mitra', 'student_id': '2023521460130', 'supervisor': 'SHAOBING GAO',
               'college': 'College of Software', 'department': 'Software Engineering', 'batch': '2023'}
    (HOME / 'experiment_report_build.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
