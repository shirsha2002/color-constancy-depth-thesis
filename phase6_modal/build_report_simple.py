r"""Simple, all-black version of the phase-6 report: only the work, results, idea and plan.

Run:  .venv\Scripts\python.exe phase6_modal/build_report_simple.py
Same numbers as build_report.py (all read from report_data); no header, footer or page numbers.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Spacer, KeepTogether, PageBreak, CondPageBreak
import build_report as R
import report_data as D

BLACK = colors.black
R.NAVY = BLACK; R.INK = BLACK; R.GREY = BLACK  # table rules and any reused colours become black
R.BODY = ParagraphStyle('b', fontName='Arial', fontSize=10, leading=14, textColor=BLACK, spaceAfter=5)
R.BULLET = ParagraphStyle('bl', parent=R.BODY, leftIndent=11, bulletIndent=2, spaceAfter=3)
R.CELL = ParagraphStyle('c', fontName='Arial', fontSize=8.4, leading=10.8, textColor=BLACK)
R.CELLB = ParagraphStyle('cb', parent=R.CELL, fontName='Arial-Bold')
H1 = ParagraphStyle('h1', fontName='Arial-Bold', fontSize=13, leading=17, textColor=BLACK, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle('h2', fontName='Arial-Bold', fontSize=10.5, leading=14, textColor=BLACK, spaceBefore=6, spaceAfter=3)
CAP = ParagraphStyle('cap', fontName='Arial', fontSize=8.5, leading=11, textColor=BLACK, spaceBefore=2, spaceAfter=9)
TITLE = ParagraphStyle('t', fontName='Arial-Bold', fontSize=17, leading=21, textColor=BLACK, spaceAfter=8)
P, B, f2, sgn, ci, table, figure = R.P, R.B, R.f2, R.sgn, R.ci, R.table, R.figure


def h(text):
    return [CondPageBreak(35*mm), P(text, H1)]


def build():
    m = lambda ds, geo, sel='fixed_final', head='released': D.pccc(ds, geo, head, sel)
    clean = m('DepthAWB', 'real')['mean']; flat_awb = m('DepthAWB', 'flat')['mean']
    nd = D.depth_conventions()['nyu_vs_diode']; bud = D.compute_budget()
    s = [P('Point Cloud Color Constancy: when does 3D information help?', TITLE),
         P(f'I trained PCCC with the authors\' code and the paper\'s full setting (10,000 epochs, official 3-fold splits) on DepthAWB (185 scenes), '
           f'ETH3D (195) and NYU-v2 + DIODE (324), on a rented Modal cloud GPU (NVIDIA T4): {bud["models"]} models, about {bud["gpu_hours"]:.1f} GPU-hours. '
           'Every model was trained twice: with the real 3D points and with the depth removed (flat). Numbers are mean angular error in degrees '
           '(lower is better) over all test scenes; "final epoch" is my main result, "best test-fold epoch" is the rule of the released code (optimistic).')]

    s += h('1  Does real 3D help?')
    rows = [['Method', 'ETH3D', 'DepthAWB', 'NYU-v2 + DIODE']]
    cell = lambda st: f'{f2(st["mean"])} ({f2(st["median"])})'
    rows.append(['Gray-World'] + [cell(D.baseline(ds, 'gray_world')) for ds in D.DATASETS])
    rows.append(['Average training illuminant'] + [cell(D.baseline(ds, 'training_mean')) for ds in D.DATASETS])
    rows.append(['PCCC flat, final epoch'] + [cell(m(ds, 'flat')) for ds in D.DATASETS])
    rows.append(['<b>PCCC real 3D, final epoch</b>'] + [f'<b>{cell(m(ds, "real"))}</b>' for ds in D.DATASETS])
    rows.append(['PCCC flat, best test-fold epoch'] + [cell(m(ds, 'flat', 'author_test_selected')) for ds in D.DATASETS])
    rows.append(['PCCC real 3D, best test-fold epoch'] + [cell(m(ds, 'real', 'author_test_selected')) for ds in D.DATASETS])
    rows.append(['Paper, w/o depth'] + [cell(D.paper(ds, 'flat')) for ds in D.DATASETS])
    rows.append(['Paper, with depth'] + [cell(D.paper(ds, 'real')) for ds in D.DATASETS])
    s.append(KeepTogether([P('Table 1. Mean (median) angular error.', H2), table(rows, [62, 36, 36, 36], zebra=False)]))
    s += [Spacer(1, 6), figure('fig1_real_vs_flat.png', 140), P('Figure 1. Real 3D vs. depth removed (final epoch). Diamonds: paper values.', CAP)]
    drows = [['Dataset', 'Real minus flat (95% CI)', 'Scenes where 3D is better']]
    for ds in D.DATASETS:
        g = D.real_minus_flat(ds); drows.append([D.NAME[ds], ci(g), f'{g["real_better"]} of {g["n"]}'])
    s.append(KeepTogether([P('Table 2. Paired difference per scene (negative = 3D better).', H2), table(drows, [40, 70, 60], zebra=False)]))
    s += [Spacer(1, 8), KeepTogether([figure('fig2_per_scene.png', 150), P('Figure 2. Every test scene; below the line, real 3D was better.', CAP)])]
    s.append(P('Real 3D clearly helps only on ETH3D. On DepthAWB and NYU-v2 + DIODE the difference is not clear, and my no-depth models are '
               'better than the paper\'s no-depth numbers on all three datasets.'))

    s += h('2  Does depth quality matter? (DepthAWB)')
    s.append(P('I damaged the real depth in four ways (10% / 30% noise, 50% of depth removed, 3D positions shuffled), then (a) retrained on it, '
               'and (b) tested the clean-trained model on it.'))
    crows = [['Depth used', '(a) Retrained', 'Change (95% CI)', '(b) Clean-trained', 'Change (95% CI)'], ['Clean', f2(clean), '–', f2(clean), '–']]
    for c in D.DEGRADED:
        crows.append([D.DEGRADED_NAME[c], f2(m('DepthAWB', c)['mean']), ci(D.degraded_minus(c, 'real')),
                      f2(D.clean_trained_on(c)['mean']), ci(D.degraded_minus(c, 'clean_trained'))])
    crows.append(['Flat (no depth)', f2(flat_awb), '', '', ''])
    s.append(KeepTogether([P('Table 3. DepthAWB, final epoch (positive change = worse).', H2), table(crows, [40, 24, 38, 30, 38], zebra=False)]))
    s += [Spacer(1, 6), KeepTogether([figure('fig3_depth_quality.png', 150), P('Figure 3. Mean error for each kind of depth.', CAP)])]
    shuf = D.degraded_minus('DepthAWB_shuffle', 'clean_trained')
    s.append(P(f'The network does use the 3D positions: shuffling them makes the clean-trained model {f2(shuf["mean"])}° worse. '
               'Noisy or missing depth at test time also hurts, so depth quality decides whether 3D is useful.'))

    s += h('3  Example scenes')
    s += [KeepTogether([figure('fig4_examples.png', 160), P('Figure 4. One test scene per dataset, chosen by ID (not by error).', CAP)]),
          KeepTogether([figure('pccc_eth3d_real_vs_flat_still.png', 150), P('Figure 5. ETH3D scene with real depth and with depth removed (see GIF).', CAP)]),
          KeepTogether([figure('pccc_depthawb_depth_damage_still.png', 128), P('Figure 6. DepthAWB scene with damaged depth (see GIF).', CAP)])]

    s += h('4  What I learned')
    s += [B('The gain from 3D is not stable: clear on ETH3D, absent on DepthAWB and NYU-v2 + DIODE.'),
          B('Depth quality matters, and PCCC cannot tell which depth values to trust.'),
          B(f'Inside NYU-v2 + DIODE the depth format is inconsistent: {nd["nyu_n"]} NYU clouds have small positive z, {nd["diode_n"]} DIODE clouds '
            f'have negative z about {nd["ratio_of_medians"]:.0f} times larger.')]
    s += h('5  My idea: depth-reliability-aware fusion')
    s += [B('Normalise depth to one sign and scale for every scene.'),
          B('Separate colour and geometry branches.'),
          B('A per-point depth-reliability gate: use geometry where depth is reliable, fall back to colour where it is not.'),
          B('Train with random depth damage so the gate learns when to trust depth.'),
          P('Goal: keep the ETH3D gain and never be worse than the no-depth model. Not implemented yet.')]
    s += h('6  Next plan')
    s += [B('Week 1: fix the NYU-v2 + DIODE depth format and rescale ETH3D depth; replace sensor depth with Depth Anything V2 on ETH3D; add two more seeds.'),
          B('Week 2: implement the fusion with ablations and test it on all three datasets and under depth damage.'),
          B('Later: multi-illuminant data and per-point illuminant estimation.')]

    # appendix statistics + references (kept, in black)
    s += [PageBreak(), P('Appendix: full statistics', H1)]
    rule = {'fixed_final': 'final epoch', 'author_test_selected': 'best test-fold'}
    arows = [['Dataset', 'Model', 'Checkpoint', 'Mean', 'Median', 'Trimean', 'Best 25%', 'Worst 25%']]
    for ds in D.DATASETS:
        for head, hn in [('released', 'released layer'), ('weighted', 'weighted layer')]:
            for geo in ['real', 'flat']:
                for sel in ['fixed_final', 'author_test_selected']:
                    st = m(ds, geo, sel, head)
                    arows.append([D.NAME[ds], f'PCCC {"real 3D" if geo == "real" else "flat"}, {hn}', rule[sel]] + [f2(st[k]) for k in D.STATS])
        for name, lab in [('gray_world', 'Gray-World'), ('training_mean', 'Average training illuminant')]:
            arows.append([D.NAME[ds], lab, '–'] + [f2(D.baseline(ds, name)[k]) for k in D.STATS])
        for geo, lab in [('flat', 'Paper, w/o depth'), ('real', 'Paper, with depth')]:
            arows.append([D.NAME[ds], lab, '–'] + [f2(D.paper(ds, geo)[k]) for k in D.STATS])
    s.append(table(arows, [29, 47, 22, 13, 14, 15, 14, 16], zebra=False))
    srows = [['Depth used (DepthAWB)', 'Model', 'Mean', 'Median', 'Trimean', 'Best 25%', 'Worst 25%']]
    for c in D.DEGRADED:
        for sel in ['fixed_final', 'author_test_selected']:
            st = m('DepthAWB', c, sel); srows.append([D.DEGRADED_NAME[c], f'retrained, {rule[sel]}'] + [f2(st[k]) for k in D.STATS])
        st = D.clean_trained_on(c); srows.append([D.DEGRADED_NAME[c], 'clean-trained, final epoch'] + [f2(st[k]) for k in D.STATS])
    s += [Spacer(1, 8), KeepTogether([P('Depth-quality test, all statistics', H2), table(srows, [44, 44, 15, 15, 16, 17, 19], zebra=False)])]
    s += [Spacer(1, 6), KeepTogether([figure('figA_eth3d_curves.png', 150), P('Training curve, ETH3D fold 0: test-fold error per epoch (thin) and 100-epoch average (thick).', CAP)])]
    s += [P('References', H1)]
    refs = ['X. Xing, Y. Qian, S. Feng, Y. Dong, J. Matas. Point Cloud Color Constancy. CVPR 2022.',
            'PCCC code and data: github.com/xyxingx/Point-Cloud-Color-Constancy',
            'C. R. Qi et al. PointNet. CVPR 2017.',
            'T. Schöps et al. ETH3D multi-view stereo benchmark. CVPR 2017.',
            'N. Silberman et al. NYU Depth V2. ECCV 2012.',
            'I. Vasiljevic et al. DIODE. arXiv:1908.00463, 2019.',
            'L. Yang et al. Depth Anything V2. NeurIPS 2024.']
    s += [P(f'[{i}] {r}', ParagraphStyle('r', parent=R.BODY, fontSize=9, leading=12, spaceAfter=2)) for i, r in enumerate(refs, 1)]

    doc = SimpleDocTemplate(str(R.OUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=18*mm,
                            title='Point Cloud Color Constancy: when does 3D information help?')
    doc.build(s)
    print(R.OUT, f'{R.OUT.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    build()
