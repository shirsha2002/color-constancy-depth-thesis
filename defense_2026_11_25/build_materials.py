"""Build the defense's authoritative content and editable written proposal."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
OUT=ROOT/'output/defense_2026_11_25'
RES=ROOT/'phase9_reliability_fusion/pilot_v1'
TITLE='Design and Implementation of a Color-Constancy-Assisted Monocular Depth Estimation System'
FOCUS='Depth Reliability Aware Fusion for Point Cloud Color Constancy'
LABELS={'rgb':'RGB only','ordinary':'Ordinary fusion','mask':'Mask fusion','learned':'Learned gate',
        'learned_clean':'Gate without damage training','learned_no_consistency':'Gate without consistency','pccc':'Original PCCC'}
ORDER=list(LABELS)
REFERENCES=[
 ('Xing X, Qian Y, Feng S, Dong Y, Matas J. Point Cloud Color Constancy. CVPR, 2022, 19750-19759.',
  'https://openaccess.thecvf.com/content/CVPR2022/html/Xing_Point_Cloud_Color_Constancy_CVPR_2022_paper.html'),
 ('Qi C R, Su H, Mo K, Guibas L J. PointNet Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR, 2017.',
  'https://arxiv.org/abs/1612.00593'),
 ('Arevalo J, Solorio T, Montes-y-Gomez M, Gonzalez F A. Gated Multimodal Units for Information Fusion. arXiv:1702.01992, 2017.',
  'https://arxiv.org/abs/1702.01992'),
 ('Wang J et al. VGGT Visual Geometry Grounded Transformer. CVPR, 2025, 5294-5306.',
  'https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html'),
 ('Wang Y et al. Pi3 Permutation-Equivariant Visual Geometry Learning. arXiv:2507.13347, 2025.',
  'https://arxiv.org/abs/2507.13347'),
 ('Keetha N et al. MapAnything Universal Feed-Forward Metric 3D Reconstruction. Official research project.',
  'https://github.com/facebookresearch/map-anything')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((RES/'audit.json').read_text())
    verification=json.loads((RES/'independent_verification.json').read_text())
    assert audit['status']==verification['status']=='PASS'
    assert all(sha(RES/n)==h for n,h in verification['audited_artifact_sha256'].items())
    summary=json.loads((RES/'summary.json').read_text()); pairs=json.loads((RES/'paired.json').read_text())
    conditions=['clean','noise10','noise30','drop50','shuffle']
    def value(method,condition):return next(r['mean'] for r in summary if r['method']==method and r['condition']==condition and r['selection']=='fixed_final')
    metrics={m:{c:value(m,c) for c in conditions} for m in ORDER}
    for m in ORDER:metrics[m]['damaged_mean']=sum(metrics[m][c] for c in conditions[1:])/4
    spec=importlib.util.spec_from_file_location('prior_report_data',ROOT/'phase6_modal/report_data.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    historical=[dict(dataset=d,label=prior.NAME[d],real=prior.pccc(d,'real')['mean'],flat=prior.pccc(d,'flat')['mean']) for d in ['ETH3D','DepthAWB','NYU_DIODE']]
    talks=[
      ('Slide 1',20,"Good afternoon. I am Shirsha Mitra. Under my registered thesis title, I am studying depth reliability aware fusion for point cloud color constancy. My current model estimates illumination from RGB and depth."),
      ('Slide 2',35,"My starting point is PCCC. In our earlier three-dataset study, real geometry helped clearly on ETH3D, but the other datasets showed no clear benefit. Damaged depth could also hurt performance. This motivates my question: can the model control how much geometry it uses when depth quality varies?"),
      ('Slide 3',40,"I implemented separate color and geometry branches. A gate uses missing-depth information and local depth disagreement to weight the geometry features. The network then estimates one global illuminant. The gate receives no test errors or corruption labels. I compare it with RGB only, ordinary fusion, mask fusion and original PCCC."),
      ('Slide 4',55,"The new ETH3D pilot trained twenty-one models for one hundred epochs across three official folds. All saved checkpoints and prediction scores passed audit. The gate's mean error was about two point five seven degrees, compared with two point five nine for mask fusion and two point three two for PCCC. Both primary uncertainty intervals include zero. Therefore, the pilot does not establish an improvement. This is preliminary evidence that guides the next experiments."),
      ('Slide 5',30,"Next I will add a learned constant-gate control, run longer schedules with repeated seeds, and test simulated scenes and additional real data. I will assess geometry from images only after verifying correspondence and illumination labels. I will also confirm alignment with the registered monocular-depth title. Thank you.")]
    payload=dict(title=TITLE,focus=FOCUS,student='Shirsha Mitra',student_id='2023521460130',university='Sichuan University',
                 preparation_date='23 November 2026',defense_date='25 November 2026',main_slides=5,backup_slides=2,
                 talk_seconds=sum(s for _,s,_ in talks),talk_word_count=len(re.findall(r"\b[\w'-]+\b",' '.join(t for _,_,t in talks))),
                 metrics=metrics,labels=LABELS,order=ORDER,historical=historical,primary=pairs,
                 script=[dict(slide=a,seconds=b,text=c) for a,b,c in talks],references=REFERENCES,
                 evidence_sha256={n:sha(RES/n) for n in ['summary.json','paired.json','audit.json','independent_verification.json','frozen_run.json']})
    (HOME/'content.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8',newline='\n')

    doc=Document(); sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.9);sec.left_margin=sec.right_margin=Cm(2)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
        style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Calibri')
    normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.1
    normal.paragraph_format.space_after=Pt(7)
    doc.styles['Title'].font.size=Pt(25);doc.styles['Heading 1'].font.size=Pt(15);doc.styles['Heading 2'].font.size=Pt(12)
    for name in ['Heading 1','Heading 2']:
        doc.styles[name].paragraph_format.space_before=Pt(10);doc.styles[name].paragraph_format.space_after=Pt(5)
    def p(text,bold=False):
        para=doc.add_paragraph();run=para.add_run(text);run.bold=bold;return para
    def h(text):doc.add_heading(text,level=1)
    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for cell,w in zip(t.rows[0].cells,widths):cell.width=Cm(w)
        for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
        for row in rows:
            for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
        for i,row in enumerate(t.rows):
            for cell,w in zip(row.cells,widths):
                cell.width=Cm(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
                tcpr.append(borders);sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EEF3' if i==0 else ('F5F7F8' if i%2==0 else 'FFFFFF'));tcpr.append(sh)
                margins=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    el=OxmlElement('w:'+edge);el.set(qn('w:w'),'80');el.set(qn('w:type'),'dxa');margins.append(el)
                tcpr.append(margins)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(2);para.paragraph_format.line_spacing=1
                    for run in para.runs:run.font.size=Pt(9);run.bold=i==0
            trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        doc.add_paragraph().paragraph_format.space_after=Pt(1)
    doc.core_properties.author='Shirsha Mitra';doc.core_properties.title='Undergraduate Thesis Proposal Report'
    # Remove inherited decorative paragraph borders from the bundled template.
    for border in list(doc.styles.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Shirsha Mitra    ')
    page_field=OxmlElement('w:fldSimple');page_field.set(qn('w:instr'),'PAGE');footer._p.append(page_field)
    for run in footer.runs:run.font.size=Pt(9)
    doc.add_paragraph('Undergraduate Thesis Proposal Report',style='Title')
    p('Research focus',True);p(FOCUS,True)
    p('Registered thesis title');p(TITLE)
    p('Sichuan University\nStudent  Shirsha Mitra\nStudent ID  2023521460130\nPrepared  23 November 2026\nProposal defense  25 November 2026')
    p('College ____________________    Major ____________________\nBatch _____________________    Supervisor ____________________')
    h('Significance of the Topic')
    p('Color constancy estimates the chromaticity of scene illumination. PCCC uses colored 3D points to supplement RGB evidence [1]. Depth can add scene structure, but missing measurements, inconsistent correspondence and geometric errors can reduce the usefulness of that information. A system that always trusts geometry may therefore make avoidable illumination errors.')
    p('I propose to investigate whether input-derived depth cues can control geometry fusion while preserving a color pathway. The immediate endpoint is global illuminant recovery from RGB-D. My registered title concerns monocular depth estimation; the present supervisor-directed PCCC study is a related module, not evidence that color constancy improves monocular depth. Formal title and scope alignment requires supervisor confirmation. Testing RGB-predicted geometry is a later connection, subject to input correspondence and label checks.')
    p('The working hypothesis is that selective geometry use can reduce degradation when depth is unreliable. The completed pilot does not yet establish that benefit. The proposal therefore emphasizes controlled comparisons, failure analysis and independent validation rather than an assumed accuracy gain.')

    doc.add_page_break();h('Literature Review')
    p('PCCC applies a PointNet-based architecture to XYZRGB clouds and extends RGB-D datasets with illumination annotations [1]. PointNet provides a way to process unordered point sets using shared point functions and symmetric aggregation [2]. These methods establish the point-cloud baseline and the need to compare real geometry with explicit controls.')
    p('Learned multimodal gates predate this project. Gated Multimodal Units learn how modalities influence intermediate features [3]. Consequently, adding a sigmoid gate or two branches alone is not sufficient novelty. My candidate must demonstrate a useful depth-quality mechanism beyond ordinary fusion, an explicit missing-depth mask, generic parameter capacity and corruption training. A learned constant gate is an important additional control for separating adaptive reliability reasoning from global geometry rescaling.')
    p('VGGT predicts scene geometry from images [4]. Pi3 develops permutation-equivariant visual geometry learning [5], while MapAnything provides a framework for metric reconstruction with several input modalities [6]. These are potential geometry sources, not results of this project. Their outputs cannot be paired with PCCC illumination labels without corresponding original images, color-processing checks and a verified image-to-point map. Multi-view geometry also changes the inference setting relative to a single-image system.')
    h('Main Research Content')
    p('First, characterize how geometry availability and damage affect illuminant estimation using the released PCCC captures. Preserve native data and distinguish measured captures from synthetic depth interventions.')
    p('Second, develop input-conditioned geometry fusion and compare it with RGB-only, ordinary fusion, mask fusion, constant gating and the original PCCC architecture under matched data and training policies.')
    p('Third, test robustness and generalization using longer schedules, repeated seeds, genuinely rendered scenes and additional real scenes with suitable illumination references. Report negative results and preserve all predeclared comparisons.')
    p('Expected outputs are reproducible data manifests, a tested implementation, an ablation study, failure analysis and a thesis report. Accuracy improvement and publishable novelty remain research outcomes to establish, not guaranteed deliverables.')

    doc.add_page_break();h('Proposed Methodology and Technical Roadmap')
    p('Each input contains 256 points with XYZ coordinates and RGB values. Separate pointwise color and geometry branches each produce 64-channel features. A small network predicts a scalar geometry weight for each valid point. Its cues are input validity, valid-depth fraction and neighboring log-depth disagreement. The neighborhood uses camera rays and includes boundary ties so point ordering does not determine the result.')
    p('Fusion rule  Fused feature = Color feature + gate x Geometry feature',True)
    p('The gate is zero for invalid geometry. Global mean and maximum context feed an illumination head. The output is one global RGB illuminant vector. The candidate has 34,020 active parameters. Original PCCC has approximately 1.68 million parameters in the pinned released architecture; this capacity difference must remain explicit.')
    p('Illumination labels supervise the output loss. They do not enter the forward gate, and neither do test errors, corruption identifiers or clean reference depth. Training combines clean and damaged-input angular losses with a prediction-consistency term. The gate is a learned mixing weight, not a calibrated probability of depth correctness. Real geometric discontinuities can also produce large disagreement cues.')
    table(['Comparison','Purpose'],[
      ['RGB only','Establish the color pathway control'],['Ordinary fusion','Test fusion without a reliability gate'],
      ['Mask fusion','Separate learned gating from simple missing-depth handling'],['Learned gate','Test the input-conditioned candidate'],
      ['Gate without damage training','Measure the contribution of corruption exposure'],['Gate without consistency','Measure the consistency objective'],
      ['Original PCCC','Retain a source-based architectural baseline'],['Learned constant gate planned','Separate adaptive cues from a global geometry weight']], [5.1,11.9])
    p('The technical sequence is input audit, fixed protocol, matched training, checkpoint re-inference, independent metric verification and error analysis. New designs and hyperparameters must use development/validation data. A genuinely independent confirmation set requires newly held-out scene identities.')

    doc.add_page_break();h('Preliminary Experiments and Findings')
    p('The earlier full-budget reproduction trained 48 models on DepthAWB, ETH3D and NYU-v2 plus DIODE, covering 704 released captures. Real geometry clearly helped on ETH3D in that experiment. The subsequent camera-frame/scale diagnostic did not establish a normalization benefit. These studies motivate the research question but do not validate the new architecture.')
    p('The new local-Intel-GPU pilot trained seven methods across three official ETH3D folds, with one seed per fold and 100 epochs per model. Each fold used 104 optimizer-training captures, 26 inner-validation captures and 65 held-out captures. Fixed epoch 100 is primary. Mean angular errors below are degrees; lower is better.')
    table(['Method','Clean','Log noise .10','Log noise .30','50% removed','XYZ shuffled'],[
      [LABELS[m]]+[f'{metrics[m][c]:.3f}' for c in conditions] for m in ORDER], [5.1,2.1,2.1,2.1,2.1,3.5])
    p('Noise uses positive lognormal depth multipliers with log-space standard deviation 0.10 or 0.30. The four damage tests are synthetic interventions on real captures, not measured sensor failures or a rendered-scene study.')
    for pair in pairs:
        lo,hi=pair['conditional_capture_bootstrap95'];endpoint='Clean inputs' if pair['endpoint']=='clean' else 'Per-capture mean over four damaged inputs'
        p(f"{endpoint}: learned gate minus mask fusion = {pair['mean_deg']:+.4f} degrees, conditional 95% interval [{lo:+.4f}, {hi:+.4f}].")
    p('Both primary intervals include zero. The learned gate did not outperform the matched original PCCC baseline. This reduced pilot therefore does not establish a gating advantage. The intervals are conditional on these models/captures, not seed or room uncertainty, and are not multiplicity adjusted.')
    p('Audit passed for all 42 checkpoints and 13,650 prediction rows. A separate scalar-angle verifier reproduced all 70 summary rows, 210 fold summaries and both primary intervals. An auditor memory-layout mismatch was corrected without changing model training or frozen inputs. The original PCCC CPU/XPU gradient discrepancy remains recorded; same-XPU repeatability passed and all pilot methods used XPU.')

    doc.add_page_break();h('Feasibility Analysis')
    p('The existing project contains 704 released labeled captures and a working local GPU implementation. The 21-model pilot establishes that the pipeline can train and be audited on the available hardware. The sparse clouds support a first controlled study even while full original image/depth correspondence is unavailable.')
    p('The main risks are limited capture/scene diversity, short training, uncertain quality cues, architecture/capacity differences and prior benchmark exposure. Mitigations are stronger matched controls, separately budgeted longer training, repeated seeds, scene-aware splits where metadata permits and new real-data confirmation. Longer training may change results, but is not guaranteed to make the candidate better.')
    p('A simulation extension should render known geometry under controlled illuminants and model sensor noise, missing depth and registration perturbations separately. Its ground truth and generation settings must be retained. New real datasets need suitable illumination labels; RGB-depth data alone do not establish a CC evaluation set. Geometry foundation models remain optional extensions until correspondence and color-space requirements are met.')
    h('Work Plan and Schedule')
    table(['Period','Planned work and output'],[
      ['23 November 2026','Finish audit, prepare proposal and timed presentation'],
      ['24 November 2026','Supervisor review if available, confirm scope/form requirements, rehearse'],
      ['25 November 2026','Attend scheduled proposal defense and record panel feedback'],
      ['Following 2 weeks','Strengthen controls and freeze the next experiment using development data'],
      ['Following 3 to 6 weeks','Budget longer schedules and repeated seeds, expand scene-aware evaluation'],
      ['Following 6 to 10 weeks','Rendered-scene tests, additional real data, method revision and thesis writing']], [4.3,12.7])
    p('The post-defense periods are planning estimates and will be revised with the supervisor. Use the local GPU where practical. Any paid cloud execution requires a concrete scope and spending limit. The earlier phase-seven eight-dollar allowance is not a new-study budget.')
    h('Expected Research Contribution')
    p('The intended contribution is a defensible mechanism for adapting geometry use to depth quality, supported by controls that isolate that mechanism. If improvement does not hold, the study will report the boundary conditions and revise the hypothesis. Neither generic gating nor an implementation-only result is sufficient evidence of a novel successful method.')

    doc.add_page_break();h('References')
    for i,(title,url) in enumerate(REFERENCES,1):p(f'[{i}] {title}\n{url}')
    h('Supporting Evidence')
    p('Project repository: https://github.com/shirsha2002/color-constancy-depth-thesis')
    p('The local source package retains the full pilot protocol, frozen manifests, every prediction vector, training histories, checkpoint identities, summary tables and independent audit. The repository is a reproducible code/evidence project rather than a complete backup of raw data and environments.')
    p('This proposal supplies the substantive sections of the university Table 3 report. The supervisor and committee retain their own assessment and signature fields. The registered title remains unchanged pending supervisor confirmation of the current RGB-D color-constancy scope.')
    doc.save(OUT/'Shirsha_Mitra_Proposal_Report.docx')

    script=['THREE MINUTE DEFENSE SCRIPT','Shirsha Mitra | 25 November 2026',f'Target: 180 seconds. Script words: {payload["talk_word_count"]}. Rehearse aloud; the timing is a target, not a measured speaking time.','']
    for name,seconds,text in talks:script.extend([f'{name} - {seconds} seconds',text,''])
    script.extend(['STOP AFTER SLIDE 5. Slides 6 and 7 are backups for questions.','',
      'বাংলায় মূল কথা',
      '১। তোমার মূল প্রশ্ন: depth-এর মান সব জায়গায় সমান নয়। Model কি input দেখে geometry কতটা ব্যবহার করবে শিখতে পারে?',
      '২। XYZ হলো অবস্থান, RGB হলো রং। Gate রং ও geometry feature মেশানোর সময় geometry-র ওজন ঠিক করে।',
      '৩। বর্তমান output হলো পুরো দৃশ্যের আলোর রং। এটি depth map নয়। Registered monocular-depth title-এর সঙ্গে এই কাজের scope স্যারের সঙ্গে নিশ্চিত করতে হবে।',
      '৪। নতুন ২১টি model-এর training ও audit শেষ। কিন্তু PCCC baseline ভালো করেছে এবং দুই primary interval-এই zero আছে। তাই improvement বা novelty প্রমাণ হয়েছে বলবে না।',
      '৫। Proposal-এ তুমি গবেষণার সমস্যা, বানানো prototype, সৎ preliminary ফল এবং পরের পরীক্ষার পরিকল্পনা দেখাবে। পুরো thesis শেষ হয়েছে বলবে না।','',
      'অনুশীলন',
      'প্রথমে প্রতিটি slide নিজের ভাষায় বোঝো। তারপর ইংরেজি script পড়ে timer চালাও। ২ মিনিট ৪৫ সেকেন্ড থেকে ৩ মিনিটের মধ্যে শেষ করার চেষ্টা করো। প্রশ্ন করলে না বুঝলে আবার বলতে অনুরোধ করো; অজানা উত্তর বানিয়ে বলবে না।'])
    (OUT/'Three_Minute_Script_English_and_Bengali.txt').write_text('\n'.join(script)+'\n',encoding='utf-8',newline='\n')
    questions=[
      ('What is color constancy?','It estimates the color of the scene illumination. Our current endpoint is a global illuminant RGB vector.','আলোর রং অনুমান করা; depth prediction নয়।'),
      ('Why use depth?','Geometry may help separate useful scene structure from ambiguous color evidence. Our earlier ETH3D comparison showed a benefit, but the benefit did not hold clearly on every dataset.','সব dataset-এ depth উপকারী হয়েছে বলা যাবে না।'),
      ('What did you implement?','Separate RGB and geometry branches, an input-derived scalar geometry gate, matched controls and a reproducible training/evaluation pipeline.','PCCC-র original code ও নিজের candidate আলাদা করে বলবে।'),
      ('Has the new algorithm improved accuracy?','Not established. The gate scored 2.574 degrees on clean ETH3D inputs versus 2.588 for mask fusion and 2.318 for PCCC. Both primary paired intervals include zero.','স্পষ্টভাবে বলবে: এখনো improvement প্রমাণ হয়নি।'),
      ('Is gating itself novel?','No. Gating is established. A contribution would need a justified depth-quality mechanism and reproducible gains beyond stronger controls and prior work.','শুধু gate যোগ করাকে নতুন গবেষণা বলবে না।'),
      ('What does the gate observe?','Input validity, valid-depth fraction and local log-depth disagreement. It does not observe test errors, corruption names or hidden clean depth.','Gate-এর value calibrated reliability নয়।'),
      ('Could a depth boundary be mistaken for bad depth?','Yes. Real depth discontinuities can raise the disagreement cues. That is a limitation and a reason for better quality supervision and boundary-aware controls.','এটি কারণ জানার সীমাবদ্ধতা; স্বীকার করবে।'),
      ('How did you prevent leakage?','Each official fold has 104 training, 26 inner-validation and 65 held-out captures in the pilot. Epoch 100 is primary. Test results did not select checkpoints or hyperparameters. Prior benchmark exposure remains development exposure.','এগুলো নতুন untouched scenes নয়।'),
      ('Why only 100 epochs?','This is a reduced-budget feasibility pilot with matched schedules. It is separate from the earlier 10,000-epoch PCCC study. We need longer schedules and repeated seeds before stronger conclusions.','পুরোনো ০.৯৮ ডিগ্রি আর নতুন pilot-এর ফল একই training budget নয়।'),
      ('Are the damage tests real sensor failures?','No. They are fixed synthetic depth interventions on real captured point clouds. Rendered-scene simulation and independent real sensor validation are still planned.','Noise যোগ করা আর আসল simulation scene তৈরি করা এক নয়।'),
      ('How does this fit the monocular-depth thesis title?','The registered title is retained. The current supervisor-directed module studies RGB-D illumination estimation. I will confirm scope alignment and test RGB-predicted geometry only with verified inputs. I do not claim an improvement in monocular depth accuracy.','এই পার্থক্য ঢাকবে না; স্যারের সঙ্গে scope ঠিক করা জরুরি।'),
      ('Did you use VGGT, Pi3 or MapAnything?','Not in these results. They are potential geometry sources for later experiments, requiring original images, correspondence and illumination-label checks.','ব্যবহার না করলে ব্যবহার করেছি বলবে না।'),
      ('Which GPU did you use?','The earlier 48-model study used Modal T4 GPUs. This new 21-model fusion pilot used my local Intel GPU. The audit used the saved checkpoints; no new Modal training was needed.','পুরোনো cloud experiment এবং নতুন local pilot আলাদা।'),
      ('What will you do next?','Add a learned constant-gate control, budget longer schedules and repeated seeds, then test genuinely simulated scenes and newly held-out real scenes.','বর্তমান test দেখে ভালো ফল আনার জন্য tuning করবে না।'),
      ('Did you write everything without assistance?','I used coding assistance to develop the implementation. I am responsible for understanding the method, checking the evidence and explaining the limitations.','নিজের না-বোঝা অংশ নিজের ভাষায় বুঝে নেবে; বানিয়ে বলবে না।')]
    qa=['DEFENSE QUESTIONS AND SHORT ANSWERS','Read these for understanding, not as invented claims.','']
    for i,(q,a,bn) in enumerate(questions,1):qa.extend([f'{i}. {q}',a,'বাংলা: '+bn,''])
    (OUT/'Defense_QA_English_and_Bengali.txt').write_text('\n'.join(qa),encoding='utf-8',newline='\n')
    checklist='''TODAY 23 NOVEMBER
1. Read slides 1 to 5 and the three-minute script.
2. Understand the output task and the inconclusive new pilot result.
3. Rehearse aloud at least three times with a timer. Target 2:45 to 3:00.
4. Read the short Q&A answers, especially novelty, negative results and title alignment.
5. Review the proposal report. Confirm blank administrative fields with your supervisor.

24 NOVEMBER
Check the PowerPoint on your own laptop. Practice screen sharing in Tencent Meeting.
Keep the slide deck and proposal locally available. Test microphone, charger and connection.
If possible, send the draft to your supervisor for review yourself.
Send only the presentation and proposal for review. Keep the private ZIP and meeting-details file private.

25 NOVEMBER
Be ready around 13:45 China time for the 14:00 session.
Presentation: 3 minutes. Questions: about 2 minutes. Stop the main talk after slide 5.
Use backups 6 and 7 only when asked. Record panel feedback.

Important: the provided notice says this defense is normally organized only once.
The proposal is a written report and presentation; a QQ message alone is insufficient.
The report follows the substantive Table 3 sections. Do not fill supervisor/committee signatures.
Today's completed package does not mean the whole thesis or a publishable new algorithm is finished.
'''
    (OUT/'START_HERE.txt').write_text(checklist,encoding='utf-8',newline='\n')
    qq='Professor, I have prepared my three-minute proposal presentation and written proposal for 25 November. I completed the 21-model ETH3D fusion pilot on my local Intel GPU and verified all 42 checkpoints and 13,650 prediction scores. The learned gate has not established an improvement over the controls, so I report the preliminary findings and limitations explicitly. My next plan includes stronger gating controls, longer schedules, repeated seeds, simulation and additional real-data validation. Could you please review the proposal focus and its alignment with my registered monocular-depth thesis title?\n'
    (OUT/'QQ_Draft_English.txt').write_text(qq,encoding='utf-8',newline='\n')
    print(json.dumps(dict(output=str(OUT),script_words=payload['talk_word_count'],talk_seconds=180,
                         proposal=str(OUT/'Shirsha_Mitra_Proposal_Report.docx')),indent=2))


if __name__=='__main__':main()
