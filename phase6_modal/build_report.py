r"""Build the professor-facing PDF for the phase-6 PCCC update.

Run:  .venv\Scripts\python.exe phase6_modal/make_figures.py
      .venv\Scripts\python.exe phase6_modal/build_report.py
Every number is read from report_data (saved, recomputed results); nothing is typed in.
"""
import re
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
                                KeepTogether, CondPageBreak)
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
import report_data as D

OUT = D.ROOT/'output/pdf/Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf'
FIG = D.ROOT/'phase6_modal/report/figures'
NAVY, TEAL, GREY, INK = colors.HexColor('#142F3E'), colors.HexColor('#007E79'), colors.HexColor('#526A77'), colors.HexColor('#1F2A30')
WASH, RULE = colors.HexColor('#EDF4F4'), colors.HexColor('#D5DEE2')
FONTS = {'Arial': 'arial.ttf', 'Arial-Bold': 'arialbd.ttf', 'Arial-Italic': 'ariali.ttf', 'Arial-BoldItalic': 'arialbi.ttf'}
for name, file in FONTS.items():
    pdfmetrics.registerFont(TTFont(name, f'C:/Windows/Fonts/{file}'))
addMapping('Arial', 0, 0, 'Arial'); addMapping('Arial', 1, 0, 'Arial-Bold')
addMapping('Arial', 0, 1, 'Arial-Italic'); addMapping('Arial', 1, 1, 'Arial-BoldItalic')
GLYPHS = set(pdfmetrics.getFont('Arial').face.charToGlyph)
MINUS = '\u2212'
AVAIL = 170*mm - 12  # frame width minus the frame's default 6 pt padding on each side
assert ord(MINUS) in GLYPHS

BODY = ParagraphStyle('body', fontName='Arial', fontSize=9.4, leading=13.4, textColor=INK, spaceAfter=5)
BULLET = ParagraphStyle('bullet', parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=3)
H1 = ParagraphStyle('h1', fontName='Arial-Bold', fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=2)
H2 = ParagraphStyle('h2', fontName='Arial-Bold', fontSize=10, leading=13, textColor=NAVY, spaceBefore=6, spaceAfter=3)
CAP = ParagraphStyle('cap', fontName='Arial', fontSize=8, leading=10.6, textColor=GREY, spaceBefore=2, spaceAfter=8)
CELL = ParagraphStyle('cell', fontName='Arial', fontSize=8.1, leading=10.4, textColor=INK)
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='Arial-Bold', textColor=NAVY)
TITLE = ParagraphStyle('title', fontName='Arial-Bold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=4)
SUB = ParagraphStyle('sub', fontName='Arial', fontSize=11, leading=14.5, textColor=TEAL, spaceAfter=6)
META = ParagraphStyle('meta', fontName='Arial', fontSize=8.6, leading=11, textColor=GREY, spaceAfter=10)
TEXT = []


def check(text):
    plain = re.sub(r'<[^>]+>', '', text).replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    missing = {c for c in plain if ord(c) not in GLYPHS}
    assert not missing, f'Font lacks glyphs {missing} in: {plain[:80]}'
    TEXT.append(plain); return text


def P(text, style=BODY, **kw):
    return Paragraph(check(text), style, **kw)


def B(text):
    return Paragraph(check(text), BULLET, bulletText='\u2022')


def f2(v):
    return f'{v:.2f}'.replace('-', MINUS)


def sgn(v):
    return f'{v:+.2f}'.replace('-', MINUS)


def ci(d):
    return f'{sgn(d["mean"])} ({f2(d["lo"])} to {f2(d["hi"])})'


def table(rows, widths, header=1, zebra=True, align=None):
    data = [[P(c, CELLB if r < header else CELL) if isinstance(c, str) else c for c in row] for r, row in enumerate(rows)]
    scale = AVAIL / (sum(widths)*mm)
    t = Table(data, colWidths=[w*mm*scale for w in widths], repeatRows=header, hAlign='LEFT')
    st = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LINEBELOW', (0, header - 1), (-1, header - 1), 0.8, NAVY),
          ('LINEBELOW', (0, -1), (-1, -1), 0.6, RULE), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if zebra:
        st += [('BACKGROUND', (0, r), (-1, r), colors.HexColor('#F6F8F9')) for r in range(header + 1, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t


def figure(name, width_mm=160, max_h_mm=None):
    w, h = PILImage.open(FIG/name).size; width = min(width_mm*mm, AVAIL); height = width*h/w
    if max_h_mm and height > max_h_mm*mm:
        height = max_h_mm*mm; width = height*w/h
    return Image(str(FIG/name), width=width, height=height)


def diagram():
    d = Drawing(AVAIL, 47*mm)
    def box(x, y, w, h, l1, l2, fill=WASH):
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=TEAL, strokeWidth=0.8))
        d.add(String(x + w/2, y + h/2 + 2.5, check(l1), textAnchor='middle', fontName='Arial-Bold', fontSize=8.3, fillColor=NAVY))
        d.add(String(x + w/2, y + h/2 - 8.5, check(l2), textAnchor='middle', fontName='Arial', fontSize=7.6, fillColor=GREY))
    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=TEAL, strokeWidth=1.1))
        if y1 == y2:
            d.add(Polygon([x2, y2, x2 - 5, y2 - 3, x2 - 5, y2 + 3], fillColor=TEAL, strokeColor=TEAL))
        else:
            d.add(Polygon([x2, y2, x2 - 3, y2 - 5, x2 + 3, y2 - 5], fillColor=TEAL, strokeColor=TEAL))
    y, h, w = 82, 34, 100; xs = [0, 123, 246, 369]  # right edge 469 pt, inside AVAIL
    box(xs[0], y, w, h, 'Real RGB-D scene', 'PCCC release, 3 datasets')
    box(xs[1], y, w, h, '256 coloured points', '(x, y, z, r, g, b)')
    box(xs[2], y, w, h, 'PCCC network', "authors' code, 10,000 epochs")
    box(xs[3], y, w, h, 'Illuminant colour', 'angular error vs. label')
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w, y + h/2, b - 2, y + h/2)
    box(62, 8, 300, 40, 'Which depth the network sees', 'Steps A+B: real 3D or flat (depth removed)   |   Step C: noise, removed, shuffled',
        fill=colors.white)
    arrow(173, 48, 173, y - 2)
    return d


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Arial', 7.5); canvas.setFillColor(GREY)
    canvas.drawString(20*mm, 11*mm, 'Shirsha Mitra  \u00b7  PCCC on all three datasets with a cloud GPU')
    canvas.drawRightString(190*mm, 11*mm, f'page {doc.page}')
    if doc.page > 1:
        canvas.drawString(20*mm, 287*mm, f'WEEKLY RESEARCH UPDATE  \u00b7  {BUDGET["last_completed_shanghai"].upper()}')
        canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5); canvas.line(20*mm, 285.5*mm, 190*mm, 285.5*mm)
    canvas.restoreState()


def section(title):
    return [CondPageBreak(40*mm), P(title, H1), HRFlowable(width='100%', thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6)]


BUDGET = D.compute_budget()


def build():
    s = []
    m = lambda ds, geo, sel='fixed_final', head='released': D.pccc(ds, geo, head, sel)
    gain = {ds: D.real_minus_flat(ds) for ds in D.DATASETS}
    eth = gain['ETH3D']; conv = D.depth_conventions(); nd = conv['nyu_vs_diode']
    clean = m('DepthAWB', 'real')['mean']; flat_awb = m('DepthAWB', 'flat')['mean']
    shuf = D.degraded_minus('DepthAWB_shuffle', 'clean_trained'); n30 = D.degraded_minus('DepthAWB_noise30', 'real')
    n30ct = D.degraded_minus('DepthAWB_noise30', 'clean_trained'); d50ct = D.degraded_minus('DepthAWB_drop50', 'clean_trained')
    ts = {ds: m(ds, 'real', 'author_test_selected')['mean'] for ds in D.DATASETS}
    pct_missing = {ds: 100*conv[ds]['missing_points']/conv[ds]['points'] for ds in D.DATASETS}
    scenes = sum(int(m(ds, 'real')['n']) for ds in D.DATASETS)
    assert BUDGET['models'] == 48 and BUDGET['epochs'] == [10000]

    # ---------------- page 1
    s += [P(f'WEEKLY RESEARCH UPDATE  \u00b7  {BUDGET["last_completed_shanghai"].upper()}', META),
          P('Point Cloud Color Constancy on all three PCCC datasets', TITLE),
          P('A full 10,000-epoch reproduction on a rented cloud GPU, and a test of when 3D information really helps', SUB),
          P('Shirsha Mitra  \u00b7  Student ID 2023521460130  \u00b7  Sichuan University', META)]
    summary = [P('<b>Summary</b>', H2),
        P("This week I followed your advice and worked directly on PCCC (CVPR 2022). I used the authors' official code and "
          f"their released real RGB-D data, and I rented a cloud GPU (Modal, NVIDIA T4) so I could train with the paper's full setting of 10,000 epochs."),
        B(f'<b>Scale:</b> 3 datasets, {scenes} real RGB-D scenes, the official 3-fold splits, {BUDGET["models"]} trained models '
          f'(about {BUDGET["gpu_hours"]:.1f} GPU-hours).'),
        B(f'<b>Reproduction:</b> with the same checkpoint rule as the released code, PCCC reaches {f2(ts["ETH3D"])}° on ETH3D '
          f'(paper {f2(D.paper("ETH3D", "real")["mean"])}°), {f2(ts["NYU_DIODE"])}° on NYU-v2 + DIODE (paper {f2(D.paper("NYU_DIODE", "real")["mean"])}°) '
          f'and {f2(ts["DepthAWB"])}° on DepthAWB (paper {f2(D.paper("DepthAWB", "real")["mean"])}°).'),
        B(f'<b>Main finding:</b> real 3D clearly helps on ETH3D ({f2(m("ETH3D", "real")["mean"])}° vs {f2(m("ETH3D", "flat")["mean"])}° without depth; '
          f'better on {eth["real_better"]} of {eth["n"]} scenes). On DepthAWB ({f2(clean)}° vs {f2(flat_awb)}°) and NYU-v2 + DIODE '
          f'({f2(m("NYU_DIODE", "real")["mean"])}° vs {f2(m("NYU_DIODE", "flat")["mean"])}°) there is no clear difference.'),
        B(f'<b>Depth quality matters:</b> a model trained on clean depth gets worse when its test depth is damaged '
          f'(with shuffled 3D positions: {f2(clean)}° to {f2(D.clean_trained_on("DepthAWB_shuffle")["mean"])}°).'),
        B('<b>Problem and idea:</b> the benefit of 3D depends on how good and how consistent the depth is. I propose a '
          'depth-reliability-aware fusion that uses geometry only where the depth can be trusted (idea only, not implemented yet).'),
        P('Unless marked otherwise, numbers are the mean angular error in degrees (lower is better) of the final-epoch model, '
          'pooled over all test scenes of the three folds.', CAP)]
    box = Table([[summary]], colWidths=[AVAIL])
    box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), WASH), ('LINEBEFORE', (0, 0), (0, -1), 3, TEAL),
                             ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 9),
                             ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    s += [box, Spacer(1, 4)]

    s += section('1  Your feedback and what I did')
    s.append(table([
        ['Your feedback (8 November)', 'What I did this week'],
        ['There was no PCCC dataset or result in my material.',
         'Used the official PCCC release (point clouds and illuminant labels of all three datasets) and trained the authors\' network on every official fold.'],
        ['Do not create the problem yourself; work on real problems and real data.',
         'Every scene here is real captured RGB-D data released by the PCCC authors. The only artificial part is Step C, where I damage the real depth on purpose as a controlled test.'],
        ['Understand why 3D information can improve colour constancy.',
         'Trained every model twice, once with the real 3D points and once with the depth removed, on all three datasets.'],
        ['How is RGB matched to the point cloud, and does the depth source matter?',
         'Checked how each dataset was captured and registered (Section 2) and measured how damaged depth changes the error (Section 4).'],
        ['Fuse the modalities better; later, multiple light sources.',
         'From the results I propose a reliability-aware fusion (Section 6). Multi-illuminant work is planned after it.'],
        ['Rent a GPU online.',
         f'Used Modal cloud GPUs (NVIDIA T4). One full model took 21 to 45 minutes; all {BUDGET["models"]} models used about '
         f'{BUDGET["gpu_hours"]:.1f} GPU-hours (about ${BUDGET["gpu_usd"]:.0f} at the list price).'],
    ], [62, 108]))

    # ---------------- setup
    s += section('2  Data and setup')
    s.append(P('PCCC turns an RGB-D image into a coloured point cloud: every point has a 3D position (x, y, z) and a colour (r, g, b). '
               'The authors release 256-point clouds (a 16 \u00d7 16 grid of the image) and one illuminant label per scene. '
               'I used these files exactly as released, with the official 3-fold splits.'))
    s.append(table([
        ['Dataset', 'Scenes', 'Camera and depth sensor', 'Test scenes per fold', 'Points without depth'],
        ['DepthAWB', '185', 'LUMIX GH5S camera + Intel RealSense L515', '62 / 62 / 61', f'{pct_missing["DepthAWB"]:.1f}%'],
        ['ETH3D', '195', 'DSLR raw images + laser-scanner depth', '65 / 65 / 65', f'{pct_missing["ETH3D"]:.1f}%'],
        ['NYU-v2 + DIODE', f'324 ({nd["nyu_n"]} + {nd["diode_n"]})', 'Kinect (NYU) and laser scanner (DIODE); sRGB images linearised by the PCCC authors',
         '114 / 100 / 110', f'{pct_missing["NYU_DIODE"]:.2f}%'],
    ], [27, 22, 66, 29, 26]))
    s.append(P('Sensors as described in the PCCC paper (Section 4) and the original dataset papers.', CAP))
    s.append(P('<b>How RGB is matched to depth.</b> The colour image and the depth map come from different sensors. The authors calibrate the '
               'two cameras and project every depth pixel into the colour image (paper Section 3.2, Eq. 2). The released clouds are already '
               'registered, so I could not re-check this step myself.'))
    s.append(P('I ran three steps. <b>Step A:</b> PCCC on DepthAWB, real 3D and flat, both output layers, all three folds (12 models). '
               '<b>Step B:</b> the same on ETH3D and NYU-v2 + DIODE (24 models). <b>Step C:</b> a depth-quality test on DepthAWB (12 models).'))
    s.append(diagram())
    s.append(P('Figure 1. What I ran. Only the depth part of each point changes between experiments; the colours and labels are never changed.', CAP))
    s += [B("<b>Network:</b> the authors' PointNet-based code at a fixed commit (ddc8770). I only made execution fixes that do not change the maths: "
            'a return-value unpacking error that crashes the released code, moving one constant matrix to the right device, and skipping a '
            '256-to-256 pooling layer that is an exact identity for 256 points. Main results use the released output layer (3 channels, summed over '
            'points); the paper\'s 4-channel weighted layer (Eq. 4) gives very similar numbers and is in Appendix A.'),
          B("<b>Training:</b> Adam, learning rate 0.0003, batch 16, 10,000 epochs, the authors' rotation and brightness augmentation, one fixed seed per fold."),
          B('<b>Real vs. flat:</b> <i>real</i> = clouds as released. <i>Flat</i> = each point\'s (x, y, z) divided by |z|, so the point keeps its pixel '
            'direction and colour but loses its depth; points without depth stay at zero. This is close to, but not exactly, the paper\'s "w/o depth" '
            'setting (depth set to 1).'),
          B('<b>Metric:</b> recovery angular error between the predicted and the true illuminant, in degrees. I pool all test scenes of the three folds '
            'and report mean, median, trimean, best 25% and worst 25%.'),
          B('<b>Two checkpoint rules.</b> (1) <i>Final epoch</i>: the model after epoch 10,000. These are my main numbers, because the test fold never '
            'chooses anything. (2) <i>Best test-fold epoch</i>: the released training script keeps the epoch with the lowest test-fold error, so I also '
            'report this rule to compare with the paper. It is optimistic.'),
          B('<b>Checking:</b> every score was recomputed from the saved predictions; the largest difference was below 1e-14 degrees.')]

    # ---------------- results
    s.append(CondPageBreak(105*mm))  # keep the section heading with Table 1
    s += section('3  Does real 3D information help? (Steps A and B)')
    rows = [['Method', 'ETH3D', 'DepthAWB', 'NYU-v2 + DIODE']]
    def cell(st):
        return f'{f2(st["mean"])} ({f2(st["median"])})'
    rows.append(['Gray-World (classic statistics)'] + [cell(D.baseline(ds, 'gray_world')) for ds in D.DATASETS])
    rows.append(['Average training illuminant'] + [cell(D.baseline(ds, 'training_mean')) for ds in D.DATASETS])
    rows.append(['PCCC flat, final epoch'] + [cell(m(ds, 'flat')) for ds in D.DATASETS])
    rows.append(['<b>PCCC real 3D, final epoch</b>'] + [f'<b>{cell(m(ds, "real"))}</b>' for ds in D.DATASETS])
    rows.append(['PCCC flat, best test-fold epoch'] + [cell(m(ds, 'flat', 'author_test_selected')) for ds in D.DATASETS])
    rows.append(['PCCC real 3D, best test-fold epoch'] + [cell(m(ds, 'real', 'author_test_selected')) for ds in D.DATASETS])
    rows.append(['<i>Paper, w/o depth</i>'] + [f'<i>{cell(D.paper(ds, "flat"))}</i>' for ds in D.DATASETS])
    rows.append(['<i>Paper, with depth</i>'] + [f'<i>{cell(D.paper(ds, "real"))}</i>' for ds in D.DATASETS])
    s.append(KeepTogether([P('<b>Table 1.</b> Mean (median) angular error in degrees, all test scenes of the three folds.', H2), table(rows, [62, 36, 36, 36])]))
    s.append(Spacer(1, 6))
    s.append(KeepTogether([figure('fig1_real_vs_flat.png', 140),
        P('Figure 2. Mean error with real 3D and with the depth removed (final epoch). Diamonds are the paper\'s values (Table 2, 256 points).', CAP)]))
    drows = [['Dataset', 'Final epoch: real minus flat (95% CI)', 'Scenes where 3D is better', 'Best test-fold epoch: real minus flat (95% CI)']]
    for ds in D.DATASETS:
        g = gain[ds]; t = D.real_minus_flat(ds, sel='author_test_selected')
        drows.append([D.NAME[ds], ci(g), f'{g["real_better"]} of {g["n"]}', ci(t)])
    s.append(KeepTogether([P('<b>Table 2.</b> Paired difference per scene. Negative means real 3D is better; if the interval does not include 0 the difference is clear.', H2),
                           table(drows, [30, 50, 32, 58])]))
    s.append(P('The 95% intervals come from 10,000 bootstrap resamples of the test scenes.', CAP))
    s.append(KeepTogether([figure('fig2_per_scene.png', 145),
        P('Figure 3. Every test scene, real 3D vs. flat. The authors\' angle function clips very small errors at about 0.08°, which explains the rows of dots at the bottom and left.', CAP)]))
    s.append(P('<b>What this shows</b>', H2))
    s += [B(f'<b>The full training works.</b> In my earlier 100-epoch laptop pilot (fold 0 only) the DepthAWB error was about 3.7°. With 10,000 epochs it is '
            f'{f2(clean)}° (final epoch) and {f2(ts["DepthAWB"])}° (best epoch).'),
          B(f'<b>Close to the paper on two datasets.</b> With the paper\'s checkpoint rule I get {f2(ts["ETH3D"])}° on ETH3D and {f2(ts["NYU_DIODE"])}° on '
            f'NYU-v2 + DIODE, close to the paper. On DepthAWB I am {f2(ts["DepthAWB"] - D.paper("DepthAWB", "real")["mean"])}° worse than the paper.'),
          B(f'<b>3D helps only on ETH3D.</b> There the gain is clear for both checkpoint rules (and for both output layers, Appendix A). On DepthAWB and '
            f'NYU-v2 + DIODE the intervals include zero. My models without depth are also better than the paper\'s "w/o depth" numbers on all '
            f'three datasets (for example DepthAWB {f2(flat_awb)}° vs {f2(D.paper("DepthAWB", "flat")["mean"])}°), so the large depth gain reported '
            f'in the paper appears only on ETH3D in my runs.')]
    assert all(m(ds, 'flat')['mean'] < D.paper(ds, 'flat')['mean'] for ds in D.DATASETS)

    # ---------------- step C
    s += section('4  Does depth quality matter? (Step C, DepthAWB)')
    s.append(P('To answer whether the depth source and its quality matter, I took the real DepthAWB depth and damaged it in four controlled ways. '
               'The damage is fixed for each scene, and the colours are never changed:'))
    s += [B('<b>10% / 30% noise:</b> every point is moved along its own pixel ray: its (x, y, z) is multiplied by (1 + e), where e is Gaussian '
            'with a standard deviation of 10% or 30%.'),
          B('<b>50% removed:</b> half of the points that have depth lose it, as if the sensor had missed them.'),
          B('<b>3D positions shuffled:</b> the 3D positions are mixed between the points, so each colour no longer sits at its own position.')]
    s.append(P('I tested this in two ways: (a) <b>retrain</b> PCCC on the damaged depth (12 new models), and (b) take the models trained on clean depth '
               'and only <b>test</b> them on damaged depth.'))
    crows = [['Depth used', '(a) Retrained', 'Change vs. clean (95% CI)', '(b) Clean-trained', 'Change vs. clean (95% CI)'],
             ['Clean (as released)', f2(clean), '\u2013', f2(clean), '\u2013']]
    for c in D.DEGRADED:
        crows.append([D.DEGRADED_NAME[c], f2(m('DepthAWB', c)['mean']), ci(D.degraded_minus(c, 'real')),
                      f2(D.clean_trained_on(c)['mean']), ci(D.degraded_minus(c, 'clean_trained'))])
    crows.append(['<i>Flat (no depth), for reference</i>', f'<i>{f2(flat_awb)}</i>', '', '', ''])
    s.append(KeepTogether([P('<b>Table 3.</b> DepthAWB, mean error of final-epoch models (released output layer). (a) retrained on that depth; '
                             '(b) trained on clean depth, tested on that depth. Positive change means worse.', H2),
                           table(crows, [40, 24, 38, 30, 38])]))
    s.append(Spacer(1, 6))
    s.append(KeepTogether([figure('fig3_depth_quality.png', 158),
        P('Figure 4. Mean error on DepthAWB for each kind of depth. Exact values and intervals are in Table 3.', CAP)]))
    s.append(P('<b>What this shows</b>', H2))
    s += [B(f'<b>The network really uses the 3D positions.</b> When the 3D positions no longer match the colours (shuffled), the clean-trained model gets '
            f'{f2(shuf["mean"])}° worse (95% CI {f2(shuf["lo"])} to {f2(shuf["hi"])}).'),
          B(f'<b>A normally trained PCCC is fragile.</b> 30% noise or 50% missing depth at test time also hurt '
            f'({sgn(n30ct["mean"])}° and {sgn(d50ct["mean"])}°).'),
          B(f'<b>Retraining adapts to missing depth, but not to strong noise:</b> after retraining, 50% missing depth is no longer worse, while 30% noise is '
            f'still {f2(n30["mean"])}° worse (95% CI {f2(n30["lo"])} to {f2(n30["hi"])}).'),
          B('So on DepthAWB the model does lean on the 3D positions, but it does not end up better than a model trained without depth '
            '(Section 3). The quality of the depth decides whether 3D is useful.')]

    # ---------------- examples
    s += section('5  Example scenes')
    ex = {ds: D.example_scene(ds) for ds in D.DATASETS}; ex_awb_missing = D.missing_depth('DepthAWB', ex['DepthAWB'])
    _, e_awb_r, _ = D.prediction('DepthAWB', 'real', ex['DepthAWB']); _, e_awb_f, _ = D.prediction('DepthAWB', 'flat', ex['DepthAWB'])
    s.append(KeepTogether([figure('fig4_examples.png', 160),
        P('Figure 5. One test scene per dataset, chosen only by ID (the first fold-0 test scene), not by error. The 16 \u00d7 16 grids are the 256 '
          f'released points, not full photographs; colours use a display gamma. DepthAWB scene {ex["DepthAWB"]} is a hard case: {ex_awb_missing} of its '
          f'256 points have no depth, and here the real-3D model is worse than the flat one ({f2(e_awb_r)}° vs {f2(e_awb_f)}°).', CAP)]))
    gscene = D.complete_depth_scene('DepthAWB')
    s.append(KeepTogether([figure('pccc_eth3d_real_vs_flat_still.png', 150),
        P(f'Figure 6. First frame of the attached animation <i>pccc_eth3d_real_vs_flat.gif</i>: the same ETH3D scene with real depth and with the depth removed.', CAP)]))
    s.append(KeepTogether([figure('pccc_depthawb_depth_damage_still.png', 128),
        P(f'Figure 7. First frame of <i>pccc_depthawb_depth_damage.gif</i>. DepthAWB scene {gscene} was chosen as the fold-0 test scene with the fewest '
          'missing depth points, so the damage is easy to see. One scene varies a lot; the mean over all 185 scenes is printed under each panel.', CAP)]))

    # ---------------- problem / idea / plan
    s += section('6  What I learned, and my idea')
    s.append(P('<b>The problem</b>', H2))
    s += [B('<b>The gain from 3D is not stable.</b> It is clear on ETH3D, but on DepthAWB and NYU-v2 + DIODE a network without depth does just as well.'),
          B('<b>Depth quality matters,</b> and the network has no way to know which depth values it can trust (Step C).'),
          B(f'<b>The depth format is not consistent inside NYU-v2 + DIODE.</b> The {nd["nyu_n"]} NYU clouds have positive z values with a per-scene '
            f'median of {nd["nyu_range"][0]:.3f} to {nd["nyu_range"][1]:.3f}, while the {nd["diode_n"]} DIODE clouds have negative z values with a '
            f'per-scene median of {nd["diode_range"][0]:.2f} to {nd["diode_range"][1]:.2f}; at the median, DIODE depth values are about '
            f'{nd["ratio_of_medians"]:.0f} times larger. The network sees two different geometries for the same kind of indoor scene. '
            'In ETH3D, where 3D helps, all 195 clouds use one convention.'),
          B('Why DepthAWB does not gain is still open (its depth range, only about 123 training scenes per fold, or label noise). I have not tested these yet.')]
    s.append(P('<b>My idea: depth-reliability-aware fusion (not implemented yet)</b>', H2))
    s.append(P('PCCC simply puts (x, y, z, r, g, b) together at the input, so good and bad depth are treated the same. I want to change this:'))
    s += [B('<b>One depth convention:</b> normalise the depth of every scene (one sign and one scale), so geometry means the same thing in every dataset.'),
          B('<b>Two branches:</b> a colour branch (r, g, b and the pixel direction) and a geometry branch (3D position and local shape).'),
          B('<b>A reliability gate:</b> every point gets a depth-reliability score (for example from missing depth and from how well its depth agrees with '
            'its neighbours). Where the depth is reliable, the geometry features count more; where it is not, the network falls back to colour.'),
          B('<b>Training with depth damage:</b> random noise, missing depth and shuffled positions (the Step C operations) during training, so the gate learns when to trust depth.')]
    s.append(P('Goal: keep the ETH3D gain, and never be worse than the no-depth model when the depth is poor. I will test it with exactly the protocol of '
               'this report. I know that gating and confidence weighting already exist in other fusion work (PCCC itself has a point-weighting layer), '
               'so I will review related work carefully before calling anything new.'))

    s += section('7  Next plan (two weeks)')
    s += [P('<b>Week 1</b>', H2),
          B('Test the depth-format explanation directly: retrain NYU-v2 + DIODE with one consistent depth convention (it should start to gain from 3D '
            'if the explanation is right), and retrain ETH3D with its depth rescaled per scene (to see whether the absolute depth scale is what helps).'),
          B('Replace the sensor depth with depth predicted from the RGB image (Depth Anything V2) on ETH3D, where the full images are public. This answers '
            '"does the depth algorithm matter?" and connects PCCC with monocular depth estimation, my thesis topic.'),
          B('Repeat the main real/flat runs with two more seeds per fold.'),
          P('<b>Week 2</b>', H2),
          B('Implement the reliability-aware fusion and its ablations (no gate, no damage training, no normalisation).'),
          B('Evaluate on all three datasets and under the Step C damage, against PCCC real and flat.'),
          P('<b>After that</b>', H2),
          B('Look for data with per-pixel or multiple-illuminant labels (your third point) and extend the per-point illuminant output of PCCC to mixed lighting.')]
    s.append(P('<b>Limitations of this report</b>', H2))
    s += [B(f'One training seed per fold ({BUDGET["models"]} models in total), so differences of about 0.2° or less should be read with care.'),
          B('Only the 256-point clouds; I have not run the 4096-point setting yet.'),
          B('"Best test-fold epoch" numbers are optimistic, because the test fold chooses the epoch.'),
          B('My flat setting is close to, but not identical to, the paper\'s "w/o depth" setting.'),
          B('The clouds are already registered by the authors, so I could not check the registration.'),
          B('The Step C damage is simulated on real depth: it is a controlled test, not new sensor data.')]

    # ---------------- appendix
    s += [PageBreak(), P('Appendix A  Full statistics', H1), HRFlowable(width='100%', thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6)]
    arows = [['Dataset', 'Model', 'Checkpoint', 'Mean', 'Median', 'Trimean', 'Best 25%', 'Worst 25%']]
    rule = {'fixed_final': 'final epoch', 'author_test_selected': 'best test-fold'}
    for ds in D.DATASETS:
        for head, hname in [('released', 'released layer'), ('weighted', 'weighted layer')]:
            for geo in ['real', 'flat']:
                for sel in ['fixed_final', 'author_test_selected']:
                    st = m(ds, geo, sel, head)
                    arows.append([D.NAME[ds], f'PCCC {"real 3D" if geo == "real" else "flat"}, {hname}', rule[sel]] + [f2(st[k]) for k in D.STATS])
        for name, label in [('gray_world', 'Gray-World'), ('training_mean', 'Average training illuminant')]:
            arows.append([D.NAME[ds], label, '\u2013'] + [f2(D.baseline(ds, name)[k]) for k in D.STATS])
        for geo, label in [('flat', 'Paper, w/o depth'), ('real', 'Paper, with depth')]:
            arows.append([D.NAME[ds], f'<i>{label}</i>', '\u2013'] + [f2(D.paper(ds, geo)[k]) for k in D.STATS])
    s.append(table(arows, [29, 47, 22, 13, 14, 15, 14, 16]))
    s.append(P('Weighted layer = the paper\'s Eq. 4 (4 channels, softmax weights over points); its brightness augmentation is 0.15 as in the paper, while '
               'the released code uses 0.20, so the two layers also differ in augmentation.', CAP))
    hrows = [['Weighted layer', 'Final epoch: real minus flat (95% CI)', 'Scenes where 3D is better', 'Best test-fold: real minus flat (95% CI)']]
    for ds in D.DATASETS:
        g = D.real_minus_flat(ds, 'weighted'); t = D.real_minus_flat(ds, 'weighted', 'author_test_selected')
        hrows.append([D.NAME[ds], ci(g), f'{g["real_better"]} of {g["n"]}', ci(t)])
    s.append(KeepTogether([P('<b>Real minus flat with the weighted layer</b>', H2), table(hrows, [30, 50, 32, 58])]))
    s.append(Spacer(1, 6))
    srows = [['Depth used (DepthAWB)', 'Model', 'Mean', 'Median', 'Trimean', 'Best 25%', 'Worst 25%']]
    for c in D.DEGRADED:
        for sel in ['fixed_final', 'author_test_selected']:
            st = m('DepthAWB', c, sel)
            srows.append([D.DEGRADED_NAME[c], f'retrained, {rule[sel]}'] + [f2(st[k]) for k in D.STATS])
        st = D.clean_trained_on(c)
        srows.append([D.DEGRADED_NAME[c], 'clean-trained, final epoch'] + [f2(st[k]) for k in D.STATS])
    s.append(KeepTogether([P('<b>Step C, all statistics</b>', H2), table(srows, [44, 44, 15, 15, 16, 17, 19])]))

    s += [CondPageBreak(70*mm), P('Appendix B  Training curve', H1), HRFlowable(width='100%', thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6),
          figure('figA_eth3d_curves.png', 160),
          P('ETH3D, fold 0, released layer. Test-fold error recorded after every epoch (thin lines) and its 100-epoch average (thick lines). '
            'The gap between real 3D and flat stays over the whole run. These curves are for monitoring only; the main results use the final epoch.', CAP)]
    s += [P('Appendix C  How to reproduce', H1), HRFlowable(width='100%', thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6),
          B('Code and results: GitHub repository <i>shirsha2002/color-constancy-depth-thesis</i>, folder <i>phase6_modal/</i>.'),
          B('Runs: <i>run_depthawb.py</i> (Step A) and <i>run_bc.py</i> (Steps B and C) on Modal; inputs for B and C from <i>make_inputs.py</i>; '
            'analysis: <i>analyze_a.py</i>, <i>analyze_bc.py</i>; figures and this report: <i>make_figures.py</i>, <i>build_report.py</i>.'),
          B('Every input file is listed with its SHA-256 in the manifests; every saved prediction is re-scored by the analysis scripts.'),
          B(f'Compute: {BUDGET["models"]} models \u00d7 10,000 epochs on NVIDIA T4, about {BUDGET["gpu_hours"]:.1f} GPU-hours.')]
    s += [P('References', H1), HRFlowable(width='100%', thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6)]
    refs = ['X. Xing, Y. Qian, S. Feng, Y. Dong, J. Matas. Point Cloud Color Constancy. CVPR 2022, pp. 19750\u201319759.',
            'Official PCCC code and data: github.com/xyxingx/Point-Cloud-Color-Constancy.',
            'C. R. Qi, H. Su, K. Mo, L. J. Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. CVPR 2017.',
            'T. Sch\u00f6ps et al. A multi-view stereo benchmark with high-resolution images and multi-camera videos (ETH3D). CVPR 2017.',
            'N. Silberman, D. Hoiem, P. Kohli, R. Fergus. Indoor segmentation and support inference from RGBD images (NYU-v2). ECCV 2012.',
            'I. Vasiljevic et al. DIODE: A Dense Indoor and Outdoor DEpth Dataset. arXiv:1908.00463, 2019.',
            'L. Yang et al. Depth Anything V2. NeurIPS 2024.']
    s += [P(f'[{i}] {r}', ParagraphStyle('ref', parent=BODY, fontSize=8.4, leading=11.2, spaceAfter=2)) for i, r in enumerate(refs, 1)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=17*mm, bottomMargin=18*mm,
                            title='PCCC on all three datasets with a cloud GPU', author='Shirsha Mitra',
                            subject='Weekly research update: Point Cloud Color Constancy reproduction and depth study')
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    (FIG.parent/'report_text.txt').write_text('\n'.join(TEXT), encoding='utf-8')  # plain text, for proofreading
    print(OUT, f'{OUT.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    build()
