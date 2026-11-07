"""Create the weekly report only from complete, independently audited results."""
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase3/results/combined'
NAVY=colors.HexColor('#142F3E');TEAL=colors.HexColor('#007E79');GREY=colors.HexColor('#526A77')
LABEL={'frozen':'Frozen model','ordinary_aug':'Ordinary + augmentation','achromatic':'Brightness control',
 'content':'Global content control','chromatic_noeq':'Global chromatic, no equivariance',
 'chromatic_eq':'Global chromatic + equivariance','flat_geometry_eq':'Global flat geometry',
 'geometry_chromatic_eq':'Global predicted geometry','spatial_content':'Local content control',
 'spatial_flat':'Local flat geometry','spatial_no_graph_loss':'Local, no graph loss',
 'spatial_no_equivariance':'Local, no equivariance','spatial_geometry_eq':'Local geometry + chromatic'}
LABEL.update({'spatial_content_noeq':'Local content, no equivariance','spatial_flat_noeq':'Local flat, no equivariance',
    'spatial_noeq_no_graph':'Local, neither added loss'})

def architecture():
    drawing=Drawing(503,155)
    def box(x,y,w,label):
        drawing.add(Rect(x,y,w,43,rx=5,ry=5,fillColor=colors.HexColor('#EDF4F4'),strokeColor=TEAL,strokeWidth=.8))
        lines=label.split('|')
        for i,line in enumerate(lines):drawing.add(String(x+w/2,y+29-i*12,line,textAnchor='middle',fontName='Arial-Bold',fontSize=8.5,fillColor=NAVY))
    def arrow(x1,y1,x2,y2):
        drawing.add(Line(x1,y1,x2,y2,strokeColor=TEAL,strokeWidth=1.2))
        if x1==x2:drawing.add(Polygon([x2,y2,x2-3,y2-6 if y2>y1 else y2+6,x2+3,y2-6 if y2>y1 else y2+6],fillColor=TEAL,strokeColor=TEAL))
        else:drawing.add(Polygon([x2,y2,x2-6,y2-3,x2-6,y2+3],fillColor=TEAL,strokeColor=TEAL))
    box(1,105,95,'One RGB|image');box(118,105,110,'Frozen encoder|shared tokens')
    box(248,105,110,'Constant task|adaptation');box(378,105,123,'Frozen decoder|refined depth')
    box(118,25,110,'Frozen first-pass|depth D0');box(248,25,110,'Predicted geometry|local affinities')
    box(378,25,123,'Local chromatic field|token residuals')
    arrow(96,126,118,126);arrow(228,126,248,126);arrow(358,126,378,126)
    arrow(173,105,173,68);arrow(228,46,248,46);arrow(358,46,378,46);arrow(439,68,439,105)
    drawing.add(String(3,5,'RGB/token context also guides the chromatic field. Measured depth is supervision/evaluation only.',fontName='Arial',fontSize=8,fillColor=GREY))
    return drawing

def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['audit']=='PASS'
    assert hashlib.sha256((OUT/'metrics.csv').read_bytes()).hexdigest()==audit['metrics_csv_sha256']
    results=json.loads((OUT/'analysis_summary.json').read_text());comparisons=json.loads((OUT/'paired_comparisons.json').read_text())
    table=results['comparison_table'];stability=results['lighting_stability'];lookup={r['method']:r for r in table}
    cfg=json.loads((OUT/'config.json').read_text());primary=cfg['primary_method']
    selected=cfg['validation_selected_method']
    for font,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(font,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    styles=getSampleStyleSheet()
    for name,font,size,leading,color in [('Body','Arial',9.6,13.5,NAVY),('Small','Arial',8.2,11,GREY),
        ('Title','Arial-Bold',25,29,NAVY),('Heading','Arial-Bold',17,22,NAVY)]:
        styles.add(ParagraphStyle(name='R'+name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=9))
    story=[]
    def p(text,style='Body'):return Paragraph(text,styles['R'+style])
    def add(text,style='Body'):story.append(p(text,style))
    def picture(path,height=None):
        width,h=ImageReader(str(path)).getSize();scale=503/width
        if height:scale=min(scale,height/h)
        story.append(Image(str(path),width=width*scale,height=h*scale));story.append(Spacer(1,8))
    def grid(rows,widths,size=8.2):
        header=ParagraphStyle('TableHeader',fontName='Arial-Bold',fontSize=size,leading=size+2,textColor=colors.white)
        body=ParagraphStyle('TableBody',fontName='Arial',fontSize=size,leading=size+2,textColor=NAVY)
        cells=[[Paragraph(escape(str(c)),header if j==0 else body) for c in row] for j,row in enumerate(rows)]
        t=Table(cells,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F5')]),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D8E3E5'))]));story.append(t);story.append(Spacer(1,10))
    def comparison(key,title):
        r=comparisons[key];lo,hi=r['bootstrap_95_interval']
        candidate='selected recipe' if '__selected_vs_' in key else 'primary'
        decimals=6 if 'difference_in_differences' in key else 4
        return f"<b>{title}:</b> {candidate} {r['candidate_mean']:.4f}, control {r['control_mean']:.4f}; paired difference {r['candidate_minus_control']:+.{decimals}f} (interval {lo:+.{decimals}f} to {hi:+.{decimals}f})."
    def newpage(title):story.append(PageBreak());add(title,'Heading')

    add('Weekly research update / 7 November 2026','Small')
    add('Real lighting and explicit chromatic controls','Title')
    add('Shirsha Mitra / Student ID 2023521460130 / Sichuan University')
    add('Design and Implementation of a Color-Constancy-Assisted Monocular Depth Estimation System','Small')
    add('This update responds to the supervisor\'s request for a clear color-constancy/depth relationship, real plus simulated evidence, expanded experiments and a method-level contribution.')
    grid([['Real source training','Captured real lighting','Simulation diagnostics'],
        ['NYU: 368 new scenes\n200 train /40 validation /128 test','ReMID: 735 real RGBs\n50 captures /25 conservative labels','Vari: 264 rendered lighting RGBs\nDigital NYU: color/exposure controls']],[167,168,168],8.5)
    picture(ROOT/'phase3/results/real_lighting_data.png',height=160)
    add('ReMID native metadata explicitly gives aligned measured depth in millimeters. We convert it to meters and keep all of ReMID external to training and checkpoint selection. Numeric scene names restart between transport shards; physical room identities are not supplied.','Small')
    add(f"<b>Completed numerical work:</b> {len(cfg['methods'])-1} trained variants plus the frozen model; {audit['prediction_rows_independently_recomputed']:,} unique evaluated predictions across the locked test inputs, plus a 640-prediction floating-point probe. Every saved score was independently recomputed. Original experiments remain separately preserved.")
    add('The exact supervisor-mentioned work is <b>Point Cloud Color Constancy, CVPR 2022</b> (Xing, Qian et al.). It predicts illumination from measured RGB-D. The new student prototype uses its own RGB-only depth prediction as geometric context. <b>LumiDepth, ECCV 2026</b> is close prior work and supplies ReMID. These relationships constrain any novelty claim.','Small')
    add('<b>Main finding:</b> the full local prototype improves digital-color AbsRel from 0.1944 to 0.1623 versus ordinary adaptation, but the local content control is slightly better at 0.1618. The validation-selected variant also transfers worse to captured ReMID than its content control. A reliable CC-specific benefit is still unproven.')

    newpage('Task-internal coupling and loss controls')
    story.append(architecture());story.append(Spacer(1,8))
    add('The local prototype predicts a relative chromatic field at encoder tokens. Normalized virtual XYZ from the frozen first-pass depth defines neighbor affinities. The field guides bounded low-rank residuals inside the depth network. The encoder and decoder remain frozen; the local module has 42,036 trainable parameters.')
    add('Depth affinity: w(i,j) = exp(-abs(log z(i) - log z(j)) /0.15). Local field smoothing and its variation penalty use the same predicted structure. These virtual coordinates are not calibrated measured world geometry.','Small')
    grid([['Comparison','Question'],['Ordinary augmented adaptation','Does additional domain fitting explain the gain?'],
        ['Brightness /generic content controls','Is the benefit explained by brightness or generic conditional capacity?'],
        ['Local flat geometry','Does predicted geometry help at the same active architecture and losses?'],
        ['No graph loss /no equivariance','Which objective contributes, and which hurts?']],[188,315])
    add('<b>Full local objective:</b> supervised AbsRel +0.1 paired log-depth agreement +0.2 local known-gain equivariance +0.01 geometry-weighted chromatic variation +0.001 residual regularization.')
    add('Every learned variant uses 600 paired updates, the same 200 NYU training scenes, augmentation schedule, AdamW recipe and validation rule. Checkpoints are chosen at updates 200,400,600 using only the forty NYU validation scenes. The local design was developed from global-model validation, before final test inspection.','Small')
    add('The cue is supervised by relative digital gain changes, not calibrated absolute illuminants. FiLM, low-rank adaptation, PointNet pooling, consistency and equivariance have established prior art. The coupling is an experimentally tested research prototype; novelty is not certified by implementation alone.','Small')

    newpage('All tested recipes: depth accuracy')
    add('All numbers below are mean AbsRel; lower is better. The same base checkpoint,224-pixel transform and fixed scoring masks apply. Per-image and conservative-group ReMID averages are both shown to expose weighting effects.','Small')
    rows=[['Method','NYU native','NYU digital color','ReMID images','ReMID groups','Vari rendered']]
    for r in table:rows.append([LABEL[r['method']],*[f"{r[k]:.4f}" for k in ['nyu_original_absrel','nyu_chromatic_absrel','remid_per_image_absrel','remid_cluster_macro_absrel','vari_rendered_absrel']]])
    grid(rows,[168,67,72,72,64,60],8.0)
    add(comparison('nyu_chromatic__primary_vs_ordinary_aug','Local primary versus ordinary augmented adaptation'))
    add(comparison('nyu_chromatic__primary_vs_spatial_content','Local primary versus local content control'))
    add(comparison('remid_cluster__primary_vs_spatial_flat','Captured real data: local primary versus flat geometry'))
    if selected!='spatial_content_noeq':add(comparison('remid_cluster__selected_vs_spatial_content_noeq','Validation-selected recipe versus loss-matched local content'))
    add(f"The validation-selected recipe is {escape(LABEL[selected])}; it was chosen from NYU validation histories before final score inspection. It is distinct from the fixed full-loss hypothesis above.",'Small')
    add('The paired intervals resample whole NYU scenes or conservative ReMID labels. ReMID intervals are conditional descriptions; unknown room identities prevent a population-independence claim. This is one training seed. The prior Vari room is exploratory and does not provide independent scene generalization.','Small')

    newpage('Color-specific effects and real-light stability')
    probe=json.loads((ROOT/'phase3/results/float_probe/summary.json').read_text())
    def probecase(name):
        r=probe['comparisons'][name]
        return {'candidate_mean':r['changed_mean_absrel'],'control_mean':r['neutral_mean_absrel'],'candidate_minus_control':r['paired_change']}
    frozen=probecase('chromatic');exposure=probecase('exposure')
    grid([['Controlled frozen-model probe','Changed AbsRel','Matched neutral','Paired change'],
        ['Chromatic changes',f"{frozen['candidate_mean']:.4f}",f"{frozen['control_mean']:.4f}",f"{frozen['candidate_minus_control']:+.4f}"],
        ['Scalar exposure change',f"{exposure['candidate_mean']:.4f}",f"{exposure['control_mean']:.4f}",f"{exposure['candidate_minus_control']:+.4f}"]],[218,95,95,95])
    drift=probe['actual_encoder_input_max_relative_luminance_drift']*100
    add(f'This separate frozen-model probe applies floating-point changes after fixed spatial resampling, with no subsequent eight-bit rounding. Actual encoder-input brightness drift is at most {drift:.6f}%. The main learned-method tables preserve their original eight-bit pipeline, whose measured drift is at most 0.6284%. Native photographs, captured lighting and rendered scenes remain separate endpoints.','Small')
    chromatic_interval=probe['comparisons']['chromatic']['bootstrap_95_interval']
    exposure_interval=probe['comparisons']['exposure']['bootstrap_95_interval']
    add(f"Paired scene-bootstrap intervals: chromatic {chromatic_interval[0]:+.4f} to {chromatic_interval[1]:+.4f}; exposure {exposure_interval[0]:+.4f} to {exposure_interval[1]:+.4f}. The chromatic interval includes zero: this probe does not establish an average color-specific error increase.",'Small')
    add(comparison('nyu_chromatic_difference_in_differences__primary_vs_spatial_content','Additional chromatic sensitivity versus local content'))
    add('The preceding difference subtracts each method\'s neutral-control error before comparing methods. It separates a response to the chromatic intervention from a general accuracy offset; lower differences favor the primary method.','Small')
    lights={(r['family'],r['method']):r for r in stability}
    rows=[['Lighting instability','Real ReMID groups','Rendered Vari views']]
    for method in ['frozen','ordinary_aug','content','spatial_content','spatial_flat','spatial_no_graph_loss','spatial_no_equivariance',primary]:
        rows.append([LABEL[method],f"{lights[('remid',method)]['conservative_cluster_macro_mad']:.4f}",f"{lights[('vari',method)]['capture_macro_mad']:.4f}"])
    grid(rows,[293,105,105])
    add(f"Instability is mean absolute prediction deviation around the fixed-capture light mean, divided by shared reference depth. It is dimensionless; lower is better. Stable but inaccurate depth is not success. All {len(cfg['methods'])} recipes and worst-deviation scores are available in the repository tables.",'Small')

    newpage('Actual saved predictions and failure evidence')
    picture(OUT/'figures/remid_depth_evidence.png',height=273)
    add('Captured ReMID example: first sorted native evaluation image. The reference is aligned measured depth; zeros and out-of-range references are masked. All depth panels share 0.1-10 m; error saturates at 2 m.','Small')
    picture(OUT/'figures/nyu_depth_evidence.png',height=273)
    add('New NYU example: first sorted native evaluation image. Selection uses no prediction-error ranking. Original Kinect-derived references include the authors\' missing-value filling. Maps are supplied for inspection and do not replace the full numerical tables.','Small')

    newpage('What remains for a paper')
    add('<b>Evidence now available:</b> real aligned multi-light RGB-D data, new scene-separated source data, explicit color/exposure interventions, two internal prototypes, controlled architecture/loss ablations and an independent prediction audit.')
    add('<b>Decision from these results:</b> do not promote the full-loss module as the winning method. The no-equivariance recipe has NYU native AbsRel 0.1510, but ReMID group AbsRel worsens to 0.4848 versus 0.4246 for loss-matched content. Small gains over flat geometry do not resolve this transfer gap. The next algorithm must address this failure rather than optimize the already-inspected test set.')
    add('<b>Limits:</b> absolute physical illuminant recovery has not been established. The prototype ingredients have close prior art. A gain over the frozen model alone is insufficient. Interpret the full content, brightness, geometry and loss controls together, including harms and tradeoffs.')
    add('Next research steps','Heading')
    add('1. Discuss the strongest matched-control comparisons with the supervisor and identify the remaining mechanism gap.<br/>2. Compare competitive published methods at matched resolution and supervision, then repeat the development recipe with independent training seeds.<br/>3. If the method changes after these results, treat this test evidence as development and reserve a new independent real dataset for confirmation.')
    add('Reproduction and code','Heading')
    add('GitHub: <link href="https://github.com/shirsha2002/color-constancy-depth-thesis" color="#007E79">github.com/shirsha2002/color-constancy-depth-thesis</link><br/>Read phase3/README.md, EXPERIMENT_V1.md, SPATIAL_EXPERIMENT.md and PAPER_PLAN.md. Raw data and large arrays remain local; source identities, code, trained adapters, tables and audit evidence are versioned.','Small')
    sources=[('Point Cloud Color Constancy (CVPR2022)','https://openaccess.thecvf.com/content/CVPR2022/html/Xing_Point_Cloud_Color_Constancy_CVPR_2022_paper.html'),
        ('LumiDepth /ReMID (ECCV2026)','https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9'),
        ('Depth Anything V2 metric implementation','https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth'),
        ('NYU Depth V2 authors','https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'),
        ('Full source and prior-art review','https://github.com/shirsha2002/color-constancy-depth-thesis/blob/main/phase3/research/LITERATURE_REVIEW.md')]
    for title,url in sources:add(f'{escape(title)}<br/><link href="{escape(url)}" color="#007E79">{escape(url)}</link>','Small')
    final=ROOT/'output/pdf/Shirsha_Mitra_Weekly_Update_2026_11_07.pdf';final.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(TEAL);canvas.setLineWidth(2);canvas.line(42,A4[1]-29,A4[0]-42,A4[1]-29)
        canvas.setFillColor(GREY);canvas.setFont('Arial',8);canvas.drawString(46,23,'SHIRSHA MITRA / RESEARCH UPDATE / 7 NOVEMBER 2026');canvas.drawRightString(A4[0]-46,23,str(doc.page))
    doc=SimpleDocTemplate(str(final),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=45,bottomMargin=42,
        title='Real lighting and explicit chromatic controls',author='Shirsha Mitra')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(final)

if __name__=='__main__':main()
