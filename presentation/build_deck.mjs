/** Build the scientific progress deck from measured pilot results only. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { p, slide, text, image, table, chart, finalize, note, C, PROJECT, ROOT } from './deck-kit.mjs';

const summary=JSON.parse(await fs.readFile(path.join(PROJECT,'results/summary.json'),'utf8'));
const figureMeta=JSON.parse(await fs.readFile(path.join(PROJECT,'results/figures/figure_manifest.json'),'utf8').catch(()=> '{}'));
const methods=['none','gray_world','white_patch','gray_world_half'];
if(summary.results.length!==4 || summary.by_condition.length!==28) throw new Error('Refusing to build from an incomplete experiment summary.');
if(summary.by_condition.some(r=>r.n_images!==summary.runtime.evaluation_images)) throw new Error('Evaluation counts differ across conditions.');
if(summary.results.some(r=>!Number.isFinite(r.corrupted_abs_rel)||!Number.isFinite(r.clean_abs_rel))) throw new Error('Nonfinite metric in summary.');
const names={none:'No correction',gray_world:'Gray-World',white_patch:'White-Patch',gray_world_half:'Gray-World half'};
const labels={clean:'Clean',warm_mild:'Warm\nmild',warm_strong:'Warm\nstrong',cool_mild:'Cool\nmild',cool_strong:'Cool\nstrong',green_mild:'Green\nmild',green_strong:'Green\nstrong'};
const results=Object.fromEntries(summary.results.map(r=>[r.method,r]));
const cond=(c,m)=>summary.by_condition.find(r=>r.condition===c&&r.method===m);
const base=results.none;
const candidate=summary.results.filter(r=>r.method!=='none').sort((a,b)=>a.corrupted_abs_rel-b.corrupted_abs_rel)[0];
const ci=summary.bootstrap.find(r=>r.method===candidate.method);
const improvement=100*(base.corrupted_abs_rel-candidate.corrupted_abs_rel)/base.corrupted_abs_rel;
const cleanChange=100*(candidate.clean_abs_rel-base.clean_abs_rel)/base.clean_abs_rel;
const intervalCrossesZero=ci.ci95_low<=0 && ci.ci95_high>=0;
const n=summary.runtime.evaluation_images;
const f=(v,d=3)=>Number(v).toFixed(d);
const percent=(v,d=1)=>`${f(100*v,d)}%`;
const generatedDate=new Date(summary.completed_at).toLocaleDateString('en-GB',{timeZone:'Asia/Shanghai',day:'numeric',month:'long',year:'numeric'});
const sourceModel='https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth';
const sourceNYU='https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html';
const sourcePaper='https://arxiv.org/abs/2406.09414';
const localSource='Measured source: results/summary.json, results/metrics.csv, data/nyu_pilot/manifest.json';
const asset=(key,fallback)=>figureMeta[key]?.path??figureMeta[key]??`results/figures/${fallback}`;
const caseExplanation=(key)=>{
  const c=figureMeta.selection?.[key];
  if(!c) return 'The example is selected after evaluation and does not represent the typical result.';
  const direction=c.displayed_absrel_reduction>=0?'reduces':'increases';
  return `Image ${c.image_id}, condition ${c.condition}. ${c.selection} In the displayed comparison, Gray-World ${direction} AbsRel by ${f(Math.abs(c.displayed_absrel_reduction),4)}. The mean six-cast AbsRel reduction for this image is ${f(c.mean_six_cast_absrel_reduction,4)}. This is a selected extreme and does not represent the typical result.`;
};

// 01. The title uses the official thesis wording, with a measured depth visual.
{
const s=slide(null,{dark:true});
text(s,'SICHUAN UNIVERSITY',64,61,750,30,{size:19,color:'#BAD0D2',bold:true});
text(s,'Design and Implementation of a',64,150,670,50,{size:29,color:'#BAD0D2'});
text(s,'Color-Constancy-Assisted\nMonocular Depth\nEstimation System',64,216,760,234,{size:51,bold:true,color:C.white});
text(s,'First controlled experiment',64,477,700,46,{size:27,color:'#77D1C5'});
text(s,'Shirsha Mitra\nStudent ID 2023521460130',64,566,700,69,{size:24,color:C.white});
text(s,generatedDate,887,612,329,35,{size:20,color:'#BAD0D2',align:'right'});
await image(s,asset('cover','cover_depth.png'),868,161,348,392,'Measured depth prediction from the pilot');
note(s,'My thesis investigates how color constancy can assist monocular depth estimation. This presentation reports the first implemented experiment. I ran a pretrained depth model and controlled color transformations on real RGB images with reference depth. The results are a local pilot, not a full benchmark reproduction or a new trained model. The figure shows an actual prediction saved by this experiment.',[sourceModel,sourceNYU,localSource]);
}
// 02. Make the research question accessible before introducing the experiment.
{
const s=slide('Research question');
text(s,'Can color correction improve depth accuracy\nwhen illumination color changes?',64,157,1152,98,{size:33,bold:true});
await image(s,asset('illumination','illumination_examples.png'),64,288,1152,285,'The same real scene under controlled original, warm, cool and green color changes');
text(s,'The scene and reference depth stay fixed. Only the input color changes.',64,605,1148,45,{size:24,color:C.muted});
note(s,'Monocular depth estimation predicts the distance of each pixel from one RGB image. Color constancy estimates or compensates for illumination color. Here I test a concrete question: if I change the input color without changing the geometry, how does the predicted depth change? Then I test whether simple correction reduces the measured error. These are synthetic global casts of real photographs. They do not represent every real lighting or camera condition.',[sourceModel,sourceNYU,localSource]);
}
// 03. A flat, editable description of the actual implemented system.
{
const s=slide('Implemented experiment',{subtitle:'One frozen depth model, paired image comparisons'});
const xs=[64,376,688,1000], ws=[270,270,270,216];
const stages=[['01','Real RGB image','Aligned reference depth\nfrom NYU Depth V2'],['02','Color processing','7 image conditions\n4 correction choices'],['03','Depth inference','Depth Anything V2\nMetric Hypersim Small'],['04','Accuracy evaluation','Same crop and mask\nNo scale fitting']];
for(let i=0;i<4;i++){
text(s,stages[i][0],xs[i],250,ws[i],47,{size:35,color:C.teal,bold:true});
text(s,stages[i][1],xs[i],318,ws[i],80,{size:28,bold:true});
text(s,stages[i][2],xs[i],414,ws[i],100,{size:24,color:C.muted});
}
text(s,`${summary.runtime.unique_predictions} saved predictions`,64,575,670,65,{size:36,bold:true,color:C.teal});
text(s,`${summary.runtime.development_images} development images and ${n} evaluation images`,704,586,512,60,{size:24,color:C.muted});
note(s,'The system has four stages. First, it loads aligned RGB and depth data. Second, it creates seven image conditions: the original, two warm casts, two cool casts, and two green casts. Each condition uses four processing choices, which gives 28 predictions per image. Third, the same pretrained model predicts depth for every input. Finally, the evaluator uses an identical crop, mask, and depth range. I save depth arrays, transformation metadata, runtime, and scores. No model training occurred.',[sourceModel,localSource]);
}
// 04. Protocol values are taken directly from the completed run.
{
const s=slide('Dataset and evaluation protocol');
const values=[['Setting','Pilot configuration'],['Data',`NYU Depth V2 RGB and aligned reference depth`],['Evaluation subset',`${n} scenes, one image each; ${summary.runtime.development_images} separate development scenes`],['Network input',`${summary.protocol.input_size} pixel inference setting, aspect ratio preserved`],['Depth evaluation','Valid reference depth 0.1-10 m, predictions clipped to this range'],['Image crop','Rows 45:471, columns 41:601 at 480 x 640'],['Aggregation','Mean of per-image scores, six casts equally weighted']];
table(s,values,64,184,1152,390,[310,842],{size:23});
text(s,'Fixed methods and gains. No parameter fitting to evaluation reference depth.',64,624,1152,36,{size:23,color:C.muted});
note(s,'This is a small deterministic pilot using real NYU Depth V2 images with aligned reference depth. Each evaluation image represents a distinct scene. Development and evaluation scenes are separate. The checkpoint predicts metric depth directly. I do not fit a scale or shift to the reference depth. The checkpoint maximum depth is 20 meters, while this pilot evaluates reference pixels strictly between 0.1 and 10 meters and clips predictions to the same evaluation range. The crop is rows 45 through 470 and columns 41 through 600 using Python slice conventions at 480 by 640. Per-image metrics are averaged. These values do not reproduce the official full benchmark. RGB and depth source details and sample selection remain in data/nyu_pilot/manifest.json.',[sourceNYU,sourceModel,localSource]);
}
// 05. Method definitions avoid implying a new algorithm contribution.
{
const s=slide('Color correction baselines',{subtitle:'All transformations operate in linear RGB'});
const data=[['Gray-World','Channel means','Estimates the illuminant\nfrom each channel mean.\nApplies inverse gains.'],['White-Patch','Channel maxima','Estimates the illuminant\nfrom each channel maximum.\nApplies inverse gains.'],['Gray-World half','Square-root gains','Uses half the correction\nstrength in log-gain space.\nStrength fixed at 0.5.']];
for(let i=0;i<3;i++){
const x=64+i*400;
text(s,data[i][0],x,242,358,80,{size:30,bold:true});
text(s,data[i][1],x,336,358,48,{size:24,color:C.teal,bold:true});
text(s,data[i][2],x,399,358,140,{size:25,color:C.muted});
}
text(s,'Every gain transform preserves mean luminance before clipping.',64,590,1152,38,{size:24,bold:true});
note(s,'I compare no correction with three fixed correction baselines. Gray-World assumes that the scene-average channel values provide useful information about the illuminant. White-Patch uses the channel maxima. Both compute inverse gains normalized with the geometric mean. The half-strength variant takes the square root of the Gray-World gains. It is a fixed ablation, not a novelty claim or a setting selected on evaluation ground truth. I convert sRGB to linear RGB before these operations, preserve mean luminance before clipping, clip the values, and convert back to sRGB. Clipping is recorded because it can remove information and make correction irreversible.',[localSource]);
}
// 06. Baseline and clean-input safeguards.
{
const s=slide('Clean-input accuracy',{subtitle:`${n} evaluation images, without an added color cast`});
const values=[['Processing','AbsRel','RMSE (m)','delta1 accuracy'],...methods.map(m=>[names[m],f(results[m].clean_abs_rel),f(results[m].clean_rmse_m),percent(results[m].clean_delta1)])];
table(s,values,64,224,1152,307,[400,220,240,292],{size:25,highlightRows:[1]});
text(s,'AbsRel and RMSE: lower is better\ndelta1: fraction of pixels within a depth ratio of 1.25, higher is better',64,567,1152,74,{size:23,color:C.muted});
note(s,`This table establishes the clean-input baseline and checks the effect of applying correction when no synthetic cast is present. The no-correction model has mean AbsRel ${f(base.clean_abs_rel,4)}, RMSE ${f(base.clean_rmse_m,4)} meters and delta-one accuracy ${percent(base.clean_delta1,2)}. AbsRel is absolute depth error divided by reference depth, averaged over valid pixels. RMSE is root mean square depth error in meters. Delta-one is the fraction of valid pixels where the larger of prediction divided by reference and reference divided by prediction is below 1.25. These are means of per-image scores on this subset. They should not be compared directly with another paper unless the dataset split and all evaluation settings match.`,[localSource]);
}
// 07. Color cast sensitivity in a native chart.
{
const s=slide('Sensitivity to color casts',{subtitle:`No correction, mean per-image AbsRel across ${n} images`});
const cs=Object.keys(summary.protocol.casts);
await chart(s,cs.map(c=>labels[c]),[{name:'No correction',values:cs.map(c=>cond(c,'none').abs_rel),fill:C.muted}],64,210,1152,392,{labels:true,yTitle:'AbsRel',axisSize:20});
text(s,'Mild and strong casts use fixed channel gains with the same reference depth.',64,622,1152,34,{size:23,color:C.muted});
note(s,'This chart measures how the unchanged model responds to the synthetic color conditions. Clean is the unmodified RGB image apart from the identity transformation and common quantization path. Warm, cool, and green casts each have mild and strong channel-gain settings. The transformation preserves mean luminance before clipping, so it does not intentionally mix a brightness experiment into this first color test. Some pixels can still clip, which is one limitation of the synthetic transformation. The exact channel gains and clipping fractions are saved in the results. A change in these scores is a measured sensitivity on this subset.',[localSource]);
}
// 08. Native mean-performance chart and paired bootstrap details.
{
const s=slide('Correction performance across color casts',{subtitle:'Fixed baselines, six synthetic casts, equal weight for each cast and image'});
await chart(s,['No\ncorrection','Gray-World','White-Patch','Gray-World\nhalf'],[{name:'AbsRel',values:methods.map(m=>results[m].corrupted_abs_rel),fill:C.teal}],64,220,770,377,{labels:true,yTitle:'Mean AbsRel',axisSize:20});
text(s,'Lowest among correction baselines',898,202,318,30,{size:18,color:C.muted});
text(s,names[candidate.method],898,244,318,69,{size:30,bold:true});
text(s,`${f(Math.abs(improvement),1)}%`,898,318,318,66,{size:49,bold:true,color:improvement>=0?C.teal:C.red});
text(s,`mean error ${improvement>=0?'reduction':'increase'}\nagainst no correction`,898,389,318,75,{size:24,color:C.muted});
text(s,`${ci.improved_images}/${ci.n_images} images improve\nwhen averaging their six casts`,898,495,318,77,{size:23,color:C.muted});
text(s,`Paired AbsRel reduction ${f(ci.mean,4)}, 95% bootstrap CI [${f(ci.ci95_low,4)}, ${f(ci.ci95_high,4)}]\n${intervalCrossesZero?'The interval crosses zero. This pilot does not establish an average benefit.':'The interval describes this pilot subset, not generalization to all scenes.'}`,64,605,1152,59,{size:20,color:C.muted});
note(s,`I average the six cast conditions for each method and each image. ${names[candidate.method]} has the lowest corrupted-input AbsRel among the three fixed correction baselines in this pilot: ${f(candidate.corrupted_abs_rel,4)}, compared with ${f(base.corrupted_abs_rel,4)} without correction. This corresponds to a ${f(Math.abs(improvement),2)} percent ${improvement>=0?'reduction':'increase'} in average AbsRel. This is a descriptive, exploratory comparison. I did not tune a model or establish a new algorithm. I also make a paired comparison per image, keeping its six casts together. The mean reduction is ${f(ci.mean,5)} with a 95 percent bootstrap interval from ${f(ci.ci95_low,5)} to ${f(ci.ci95_high,5)}. A positive reduction means correction helped. A negative reduction means it harmed. I resample images 10,000 times. ${ci.improved_images} images improve and ${ci.harmed_images} worsen. This interval describes the sampled images and does not establish generalization to other datasets or all real lighting.`,[localSource]);
}
// 09. One actual example, with all labels embedded in the scientific figure.
{
const s=slide('Visual comparison of saved depth predictions');
await image(s,asset('best','qualitative_best.png'),64,167,1152,443,'Actual RGB inputs, reference depth and measured predictions for a selected evaluation example');
text(s,'The reference and predicted depth maps share the same color scale.',64,622,1152,32,{size:23,color:C.muted});
note(s,'This figure shows an actual evaluated image and the corresponding saved depth arrays. '+caseExplanation('best')+' The reference depth does not change across color conditions. Compare object boundaries and relative near/far regions, and use the numerical error labels to assess accuracy. A visually smooth prediction can still have an incorrect metric scale. The preceding aggregate results provide the performance estimate.',[localSource]);
}
// 10. Present a measured failure, rather than implying universal improvement.
{
const s=slide('Failure analysis');
await image(s,asset('failure','qualitative_failure.png'),64,164,1152,425,'Actual evaluated scene selected for the weakest average Gray-World effect');
text(s,'A global correction can mistake scene color for illumination.',64,603,1152,40,{size:27,bold:true,color:C.red});
note(s,'I also inspect the image with the weakest average Gray-World effect across the six casts. '+caseExplanation('failure')+' A global illuminant estimator can confuse the color of the scene with the color of the light. A scene dominated by one surface color can violate Gray-World assumptions. Channel maxima can also make White-Patch unstable. Clipping can discard color information before correction. These are possible mechanisms to investigate with further controlled tests, not a proven causal diagnosis of this single example. The clean-input comparison and per-image paired results show why applying correction to every input needs further study.',[localSource]);
}
// 11. Actual execution and files make the progress reviewable.
{
const s=slide('Laptop execution and reproducibility');
const values=[['Item','Completed run'],['Device',`${summary.runtime.cpu}, CPU inference`],['Model',`${summary.protocol.model}, ${summary.protocol.parameters} parameters`],['Compute setting',`${summary.runtime.threads} CPU threads, input setting ${summary.protocol.input_size}`],['Predictions',`${summary.runtime.unique_predictions} numerical depth arrays saved`],['Mean inference time',`${f(summary.runtime.mean_inference_seconds,2)} seconds per image condition`],['Traceable outputs','Manifest, metrics CSV, checkpoint hash, figures and source code']];
table(s,values,64,186,1152,393,[330,822],{size:23});
text(s,'The project runs in its own Python environment and opens as a VS Code workspace.',64,610,1152,34,{size:23,color:C.muted});
note(s,`The experiment ran locally using CPU inference on an ${summary.runtime.cpu} with approximately ${summary.runtime.ram_gb} GB RAM. The run used ${summary.runtime.threads} CPU threads. Mean measured inference time is ${f(summary.runtime.mean_inference_seconds,3)} seconds per transformed input. It measures the depth inference call and excludes model loading and artifact preparation. The input-size parameter is ${summary.protocol.input_size}. I save numerical predictions, per-image scores, an aggregate summary, source code, the data manifest, the upstream revision, and the checkpoint SHA-256. The full hash is ${summary.protocol.checkpoint_sha256}. The upstream model revision is ${summary.protocol.upstream_git_revision}. These files support rerunning and auditing the experiment.`,[sourceModel,localSource]);
}
// 12. Findings are deliberately scoped to the observed pilot.
{
const s=slide('Findings and next experiments',{dark:true});
text(s,'Observed in this pilot',64,195,510,48,{size:29,bold:true,color:'#77D1C5'});
text(s,`The complete paired evaluation runs locally.\n\n${names[candidate.method]} has ${f(Math.abs(improvement),1)}% ${improvement>=0?'lower':'higher'} mean cast error.\n\nIts mean clean-input error is ${f(Math.abs(cleanChange),1)}% ${cleanChange>=0?'higher':'lower'}.`,64,267,540,305,{size:27,color:C.white});
text(s,'Next experiments',713,195,503,48,{size:29,bold:true,color:'#77D1C5'});
text(s,'Expand the fixed evaluation subset and test a second depth model.\n\nSeparate clipping effects from color sensitivity.\n\nTest real lighting changes before designing an adaptive correction rule.',713,267,503,305,{size:27,color:C.white});
text(s,'Initial pretrained-model pilot. A novel method and broader validation remain future work.',64,626,1152,35,{size:20,color:'#BAD0D2'});
note(s,`The current progress is a working system and a controlled first experiment. This descriptive pilot suggests what to investigate next. Compared with no correction, the lowest-error correction baseline has ${f(Math.abs(improvement),2)} percent ${improvement>=0?'lower':'higher'} average cast AbsRel, and ${f(Math.abs(cleanChange),2)} percent ${cleanChange>=0?'higher':'lower'} clean-input AbsRel. Both comparisons use the same fixed model and evaluation images. Neither result justifies a universal claim about color correction or a novel-method claim. My next steps are a larger fixed subset, another depth backbone, a separate clipping control, and images captured under real illumination changes. Only after these checks should I decide whether an adaptive correction method is justified. I would welcome the supervisor's guidance on the most useful next comparison.`,[sourcePaper,sourceModel,sourceNYU,localSource]);
}

const notes=p.slides.items.map((s,i)=>`SLIDE ${i+1}\n${s.speakerNotes.text}`).join('\n\n'+'='.repeat(72)+'\n\n');
await fs.writeFile(path.join(PROJECT,'output','Shirsha_Mitra_Presentation_Notes.txt'),notes,'utf8');
const completed=await finalize(`Shirsha_Mitra_Thesis_Progress-build-${Date.now()}`,{
  explicitTotalSlideCount:12,
  requiredNativeTableOwnerSlides:[4,6,11],
  requiredNativeChartOwnerSlides:[7,8],
},'Shirsha_Mitra_Thesis_Progress.pptx');
await fs.writeFile(path.join(ROOT,'.build/latest_build.json'),JSON.stringify({
  completedAt:new Date().toISOString(),finalPath:completed.deliveredPath,
  sha256:completed.finalSha256,validationReceipt:completed.receiptPath,slides:12,
},null,2),'utf8');
console.log('Deck and presenter notes completed');

