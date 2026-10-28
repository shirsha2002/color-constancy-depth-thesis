"""Create a concise, fully measured phase-two research note and teacher reply."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

ROOT=Path(__file__).resolve().parents[1];MONO=ROOT/'phase2/results/mono'
NAVY=colors.HexColor('#132D3C');TEAL=colors.HexColor('#007F7B');GREY=colors.HexColor('#576A76')
NAMES={'frozen':'Frozen original','gray_world_preprocess':'Gray-World preprocessing','task_adapter':'Task adapter (no color cue)',
       'achromatic_consistency':'Achromatic + consistency','cc_supervised':'Chromatic / depth only','cc_consistency':'Chromatic + consistency'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stereo-summary',type=Path,required=True);args=parser.parse_args()
    s=json.loads((MONO/'summary.json').read_text());st=json.loads(args.stereo_summary.read_text())
    if not st.get('passed'):raise ValueError('Stereo numerical audit has not passed')
    results={r['method']:r for r in s['results']};cc=results['cc_consistency'];task=results['task_adapter'];frozen=results['frozen']
    audit=json.loads((MONO/'independent_audit.json').read_text())
    if audit.get('audit')!='PASS':raise ValueError('Mono numerical audit has not passed')
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2',fontName='Arial',fontSize=10,leading=14,textColor=NAVY,spaceAfter=8))
    styles.add(ParagraphStyle(name='Title2',fontName='Arial-Bold',fontSize=25,leading=30,textColor=NAVY,spaceAfter=15))
    styles.add(ParagraphStyle(name='Heading2x',fontName='Arial-Bold',fontSize=17,leading=22,textColor=NAVY,spaceAfter=12))
    styles.add(ParagraphStyle(name='Small2',fontName='Arial',fontSize=8.5,leading=11,textColor=GREY,spaceAfter=6))
    story=[]
    def p(text,style='Body2'):return Paragraph(text,styles[style])
    def add(text,style='Body2'):story.append(p(text,style))
    def picture(path,width=493,height=None):
        im=ImageReader(str(path));w,h=im.getSize();scale=width/w
        if height and h*scale>height:scale=height/h
        story.append(Image(str(path),width=w*scale,height=h*scale));story.append(Spacer(1,8))
    def table(rows,widths):
        data=[[p(str(c),'Small2') for c in row] for row in rows];t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F4')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D9E2E5'))]))
        # Paragraph colors override Table textColor, so style header paragraphs directly.
        for col in range(len(rows[0])):data[0][col]=Paragraph(escape(str(rows[0][col])),ParagraphStyle(name='header'+str(col),fontName='Arial-Bold',fontSize=8.5,leading=11,textColor=colors.white))
        t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F4')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D9E2E5'))]));story.append(t);story.append(Spacer(1,11))
    add('Research update after supervisor feedback','Small2')
    add('Color constancy inside geometry estimation','Title2')
    add('Shirsha Mitra  /  Student ID 2023521460130  /  29 October 2026')
    add('This phase acquires matched multi-light data, measures lighting sensitivity, and implements a small CC-inspired module inside a depth model. It preserves the first NYU preprocessing study as a separate baseline.')
    table([['Completed action','Evidence'],['Dataset search and acquisition','Vari: 528 RGB images, 48 fixed-camera groups, 11 lighting conditions, 2 rendered rooms. DTU: 42 real RGB images, 2 scenes, calibrated cameras and structured-light reference surfaces.'],['Task-integrated prototype','101,504 trainable parameters between the frozen DINOv2 encoder and frozen depth decoder. Raw RGB remains unchanged. Depth and paired-light consistency train the internal features.'],['Controlled comparisons','Frozen model, image preprocessing, ordinary task adaptation, achromatic conditioning, chromatic conditioning, and consistency ablation.'],['Current interpretation','Lighting affects geometry. The present chromatic module does not outperform ordinary task adaptation; a CC-specific gain is not established.']],[130,363])
    picture(MONO/'figures/lighting_inputs.png',height=150)
    add('Data limits: Vari publicly releases only its test archive; we create a new exploratory split, with room 212 for development and room 485 for evaluation. These are not official benchmark scores. DTU varies lighting direction and does not supply ground-truth illuminant colors.','Small2')
    story.append(PageBreak())
    add('Monocular depth experiment','Heading2x')
    add('Development: 20 training and 4 validation camera groups in room 212. Evaluation: 24 camera groups in room 485, 264 lighting images. The evaluation contains one unseen room, not 24 independent scenes. All methods use the same 224-pixel input transform, reference pixels and frozen depth backbone.')
    rows=[['Method','AbsRel','RMSE (m)','Delta1','Light MAD']]
    for m in s['results']:rows.append([NAMES[m['method']],f"{m['abs_rel']:.4f}",f"{m['rmse_m']:.3f}",f"{100*m['delta1']:.1f}%",f"{m['lighting_instability']:.4f}"])
    table(rows,[205,65,78,65,80])
    picture(MONO/'figures/mono_comparison.png',height=225)
    change=100*(cc['abs_rel']/task['abs_rel']-1)
    add(f"The chromatic consistency model has {change:.1f}% higher AbsRel than the ordinary task adapter ({cc['abs_rel']:.4f} versus {task['abs_rel']:.4f}). Its large improvement over the unadapted model is therefore not evidence that CC itself caused the gain.")
    add('AbsRel and RMSE are means of per-image scores. Light MAD is the mean absolute deviation across eleven lighting predictions around their per-pixel mean, divided by reference depth. Lower is better for these three errors; higher Delta1 is better. Accuracy and stability must be interpreted together.','Small2')
    story.append(PageBreak())
    add('Saved depth predictions and ablations','Heading2x')
    picture(MONO/'figures/depth_evidence.png',height=310)
    add('The example is the first sorted evaluation camera and lighting condition, chosen without ranking model performance. All depth maps use the same 0.1-10 m color scale. The error map saturates at 2 m. The ordinary adapter is an essential control for general domain/scale adaptation.','Small2')
    picture(MONO/'figures/lighting_accuracy.png',height=210)
    add('Training uses 150 paired updates per learned variant, AdamW and train-only descriptor statistics. Validation selects checkpoints only in room 212. Primary references preserve sixteen-bit TIFF depth; the authors-compatible eight-bit path is retained and checked separately. No prediction-specific scale fitting occurs.','Small2')
    story.append(PageBreak())
    add('Calibrated real-lighting geometry study','Heading2x')
    add('DTU uses real images and original structured-light surfaces. The same calibrated view pair is tested under matched lighting and a changed source light. Plane-sweep matching estimates depth directly; radiometric gains are fitted inside each depth-hypothesis cost, rather than writing corrected RGB images.')
    add('The final stereo results and numerical checks are available in phase2/results/stereo/main_v2. Methods include RGB SSD, scalar gain, diagonal gain, grayscale ZNCC and per-channel ZNCC. The known robust costs are strong baselines; the gain-based prototype is not a novelty claim.')
    # Filled by the combined handoff using the stereo agent's normalized result table.
    normalized=st.get('report_table')
    if not normalized:raise ValueError('Stereo summary must provide verified report_table and report_interpretation')
    table(normalized,[173,105,110,105])
    add(st['report_interpretation'])
    if st.get('report_figure'):picture(ROOT/st['report_figure'],height=255)
    add('Units are millimeters. Scene 1 supports geometry checks; scene 6 is held out. Masks and depth planes are shared across methods. Reference surfaces enter evaluation and visibility checking, never matching costs. One held-out scene cannot support a population-level confidence claim.','Small2')
    story.append(PageBreak())
    add('What this phase establishes','Heading2x')
    add('<b>Completed:</b> supporting datasets were acquired and audited; geometry was tested under matched lighting variants; a trainable CC-inspired feature module now participates inside depth inference; strong task-adaptation and lighting-invariance controls were evaluated.')
    add('<b>Remaining gap:</b> this prototype has not demonstrated an advantage attributable specifically to CC. The unadapted model has a large domain/scale bias on Vari, and the ordinary task adapter remains stronger than the current chromatic version. Two rooms are insufficient for broad learning claims.')
    add('<b>Next research decision:</b> obtain more diverse scene-separated paired-light data; test cues less dependent on average room color, such as neutral-surface or relative-color evidence; retain achromatic and ordinary-adaptation controls. Improve the method on development data before evaluating another independent scene set.')
    add('Reproduction and evidence','Heading2x')
    add('Project: D:/thesis work/thesis code<br/>Branch: research/phase2-integrated-cc<br/>GitHub: https://github.com/shirsha2002/color-constancy-depth-thesis/tree/research/phase2-integrated-cc','Small2')
    add('Fresh monocular reproduction:<br/>.venv/Scripts/python.exe phase2/audit/reproduce_guard.py --run-fresh --out phase2/results/mono_reproduction_01<br/>The destination must not exist. The guard verifies frozen inputs and prevents stale cache reuse. See phase2/README.md for data acquisition, audits and stereo settings.','Small2')
    sources=[('Depth Anything V2 metric implementation','https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth'),('Vari dataset and DCA authors','https://github.com/zzt76/dca'),('DTU original data and calibration','https://roboimagedata.compute.dtu.dk/?page_id=36'),('FiLM feature modulation background','https://arxiv.org/abs/1709.07871')]
    for title,url in sources:add(f'{escape(title)}<br/><link href="{escape(url)}" color="#007F7B">{escape(url)}</link>','Small2')
    out=ROOT/'output/pdf/Shirsha_Mitra_Phase2_Research_Update.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(TEAL);canvas.setLineWidth(2);canvas.line(43,A4[1]-31,A4[0]-43,A4[1]-31)
        canvas.setFont('Arial',8);canvas.setFillColor(GREY);canvas.drawString(43,24,'SHIRSHA MITRA / PHASE TWO RESEARCH UPDATE');canvas.drawRightString(A4[0]-43,24,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=51,rightMargin=51,topMargin=49,bottomMargin=44,title='Color constancy inside geometry estimation',author='Shirsha Mitra')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(out)

if __name__=='__main__':main()
