import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const HOME=path.dirname(fileURLToPath(import.meta.url));
const ROOT=path.dirname(HOME);
const OUT=path.join(ROOT,'output/defense_2026_11_25');
const BUILD=path.join(HOME,'.build');
const SKILL='C:/Users/Shirshas Laptop/.codex/plugins/cache/openai-primary-runtime/presentations/26.1118.11504/skills/presentations';
const RUNTIME='C:/Users/Shirshas Laptop/.cache/codex-runtimes/codex-primary-runtime';
const data=JSON.parse(await fs.readFile(path.join(HOME,'content.json'),'utf8'));
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={paper:'#FAFAF7',ink:'#18313F',muted:'#526575',teal:'#007C78',blue:'#3E7196',gray:'#8E9DA5',line:'#D7E0E3'};
const FONT='Arial';
function text(s,value,x,y,w,h,size=30,opt={}){
 const t=s.shapes.add({name:opt.name??value.slice(0,40),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:FONT,fontSize:size,color:opt.color??C.ink,bold:opt.bold??false,alignment:'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};return t;
}
function slide(title,backup=false){
 const s=p.slides.add();s.background.fill=C.paper;
 if(title)text(s,title,64,48,1152,100,44,{bold:true});
 text(s,backup?'Backup '+(p.slides.items.length-5):String(p.slides.items.length),1160,667,65,28,19,{color:C.muted});
 return s;
}
function note(s,index,extra=''){
 s.speakerNotes.text=(index<5?`${data.script[index].seconds} seconds\n${data.script[index].text}`:'Backup only. Open this slide if the panel asks for details.')+
  '\n\n'+extra+'\n\nSources\n'+data.references.map((r,i)=>`[${i+1}] ${r[0]} ${r[1]}`).join('\n')+
  '\nLocal pilot evidence: phase9_reliability_fusion/pilot_v1/summary.json, paired.json, audit.json and independent_verification.json. Historical evidence: phase6_modal/results. Audit completed 23 November 2026.';
}
function chart(s,categories,series,x,y,w,h){
 const ch=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,
  series:series.map((r,i)=>({name:r.name,values:r.values.map(v=>Number(v.toFixed(8))),fill:i?C.teal:C.gray,line:{fill:i?C.teal:C.gray,width:1}})),
  hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:C.ink}},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
  xAxis:{textStyle:{fontSize:23,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},
  yAxis:{min:0,title:{text:'Mean angular error (degrees)',textStyle:{fontSize:22,fill:C.ink}},
         numberFormatCode:'0.0',textStyle:{fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:0.7}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:23,fill:C.ink}},
  chartFill:C.paper,chartLine:{fill:'none',width:0},plotAreaFill:C.paper,plotAreaLine:{fill:'none',width:0}});
 applyPresentationChartFont(ch,{fontFamily:FONT});return ch;
}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({style:'solid',fill:C.line,width:.6});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.paper,textStyle:{typeface:FONT,fontSize:23,color:C.ink},margins:{top:12,bottom:10,left:10,right:10},anchor:'center'});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.ink,textStyle:{typeface:FONT,fontSize:23,bold:true,color:'#FFFFFF'}});
 return t;
}

let s=slide('');
text(s,'Undergraduate proposal defense',64,55,1120,48,26,{color:C.teal});
text(s,'Design and Implementation of a\nColor-Constancy-Assisted Monocular\nDepth Estimation System',64,145,1152,230,56,{bold:true});
text(s,'Research focus\nDepth reliability aware fusion for point cloud color constancy',64,410,1136,110,30,{color:C.muted});
text(s,'Shirsha Mitra\n2023521460130   Sichuan University',64,560,1050,76,27);
text(s,'25 November 2026',64,657,750,30,23,{color:C.muted});note(s,0,'Registered title is retained. Current PCCC module estimates illumination from measured RGB-D. Confirm title/scope alignment with the supervisor.');

s=slide('Geometry helps on some datasets');
chart(s,['ETH3D','DepthAWB','NYU + DIODE'],[
 {name:'Native XYZ',values:data.historical.map(r=>r.real)},
 {name:'Ray-flat control',values:data.historical.map(r=>r.flat)}],52,170,825,395);
text(s,'ETH3D\n0.98 vs 1.31 degrees',900,193,315,100,31,{bold:true,color:C.teal});
text(s,'The other datasets show no clear geometry benefit.\n\nDepth quality is a hypothesis to investigate.',900,335,315,222,27);
text(s,'Historical PCCC study: 10,000 epochs, official folds, fixed-final checkpoints',64,599,1150,44,23,{color:C.muted});
text(s,'Flat geometry preserves camera rays and missingness',64,642,1080,30,21,{color:C.muted});note(s,1,'These are project reproduction results, not the paper-published numbers. Real-minus-flat conditional intervals include zero on DepthAWB and NYU/DIODE.');

s=slide('Candidate reliability aware fusion');
text(s,'Input: 256 XYZRGB points. Output: one global illuminant RGB vector.',64,149,1150,55,27,{color:C.muted});
text(s,'Color branch\nRGB appearance features',64,246,520,110,35,{bold:true,color:C.teal});
text(s,'Geometry branch\nXYZ structure features',700,246,510,110,35,{bold:true,color:C.blue});
text(s,'g = validity x sigmoid(MLP(depth cues))',64,397,1150,64,40,{bold:true});
text(s,'Fused feature = Color feature + g x Geometry feature',64,474,1150,55,32);
text(s,'Gate cues: missing depth and local log-depth disagreement\nThe gate receives no test errors or corruption labels',64,572,1150,76,26,{color:C.muted});
note(s,2,'The gate uses input validity, valid fraction, mean and RMS neighboring log-depth differences. It has no calibrated reliability ground truth. Valid depth discontinuities may also raise these cues. Candidate parameters: 34,020.');

s=slide('Pilot results remain inconclusive');
chart(s,['PCCC','Mask fusion','Learned gate'],[
 {name:'Clean',values:['pccc','mask','learned'].map(m=>data.metrics[m].clean)},
 {name:'Damaged mean',values:['pccc','mask','learned'].map(m=>data.metrics[m].damaged_mean)}],52,177,790,390);
text(s,'Learned gate minus mask',880,184,335,65,28,{bold:true});
text(s,'Clean\n-0.014 degrees\n95% interval [-0.117, +0.085]',880,275,335,110,25);
text(s,'Damaged mean\n-0.090 degrees\n95% interval [-0.183, +0.002]',880,425,335,120,25);
text(s,'21 models, 100 epochs, 195 captures, three folds and one seed per fold',64,593,1150,38,23,{color:C.muted});
text(s,'Both primary intervals include zero. PCCC performed better in this pilot.',64,637,1150,38,25,{bold:true});
note(s,3,'Damaged mean is the per-capture average over four fixed tests: lognormal depth noise SD 0.10/0.30, removal of half valid geometry, XYZ shuffle. Intervals condition on these captures/models, do not represent seed/room uncertainty, and are not multiplicity adjusted. All 42 checkpoints and 13,650 rows passed audit.');

s=slide('Research plan and scope');
text(s,'1   Stronger controls',64,190,1150,50,35,{bold:true,color:C.teal});
text(s,'Learned constant gate and capacity-matched alternatives',104,250,1100,52,29);
text(s,'2   Stronger evidence',64,335,1150,50,35,{bold:true,color:C.teal});
text(s,'Longer schedules, repeated seeds and newly held-out scenes',104,395,1100,52,29);
text(s,'3   Broader evaluation',64,480,1150,50,35,{bold:true,color:C.teal});
text(s,'Rendered scenes and additional real data with illumination labels',104,540,1100,52,29);
text(s,'Confirm alignment with the registered monocular-depth title',64,636,1150,40,25,{color:C.muted});
note(s,4,'Stop the timed presentation here. Geometry foundation models are optional later inputs, requiring original images and verified correspondence. The completed pilot does not establish novelty or generalization.');

s=slide('All pilot methods and damage tests',true);
const short={rgb:'RGB only',ordinary:'Ordinary fusion',mask:'Mask fusion',learned:'Learned gate',learned_clean:'Gate, clean training',learned_no_consistency:'Gate, no consistency',pccc:'Original PCCC'};
table(s,[['Method','Clean','Noise .10','Noise .30','Drop .50','Shuffle'],...data.order.map(m=>[short[m],...['clean','noise10','noise30','drop50','shuffle'].map(c=>data.metrics[m][c].toFixed(3))])],64,170,1152,420,[342,162,162,162,162,162]);
text(s,'Mean angular error in degrees, lower is better. Fixed epoch 100.',64,610,1150,34,24,{color:C.muted});
text(s,'Noise is lognormal depth scaling. These are synthetic interventions on real clouds.',64,651,1150,30,21,{color:C.muted});note(s,5);

s=slide('Scope and validation',true);
text(s,'Completed',64,181,1110,50,34,{bold:true,color:C.teal});
text(s,'PCCC reproduction, fusion implementation and audited ETH3D pilot\n42 saved checkpoints and 13,650 prediction scores verified',64,243,1140,108,29);
text(s,'Still to establish',64,385,1110,50,34,{bold:true,color:C.teal});
text(s,'Reliable performance improvement and algorithm novelty\nRepeated-seed and independent real-scene generalization\nA verified connection to the registered monocular-depth task',64,447,1140,145,29);
text(s,'Synthetic depth damage does not replace rendered-scene or real sensor validation',64,635,1150,40,23,{color:C.muted});note(s,6,'Prior work includes PCCC [1], PointNet [2] and gated multimodal units [3]. VGGT [4], Pi3 [5] and MapAnything [6] are possible later geometry sources, not methods used in the present results.');

await fs.mkdir(BUILD,{recursive:true});await fs.mkdir(OUT,{recursive:true});
const candidate=path.join(BUILD,'defense-final.candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(BUILD,'validated/defense-final4.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const receipt=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,
 pythonExecutable:path.join(RUNTIME,'dependencies/python/python.exe'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','6'],
 explicitTotalSlideCount:7,requiredNativeChartOwnerSlides:[2,4],requiredNativeTableOwnerSlides:[6],
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,materializeLiteralChartWorkbooks:true,
 receiptPath:path.join(BUILD,'validation-final4.json')});
for(const [i,sl] of p.slides.items.entries()){
 const png=await p.export({slide:sl,format:'png',scale:1.25});
 await fs.writeFile(path.join(BUILD,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
await fs.writeFile(path.join(BUILD,'build_receipt.json'),JSON.stringify(receipt,null,2));
await fs.copyFile(finalPath,path.join(OUT,'Shirsha_Mitra_Proposal_Defense_2026.pptx'));
console.log(JSON.stringify(receipt));
