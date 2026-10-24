import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const ROOT = path.dirname(fileURLToPath(import.meta.url));
export const PROJECT = path.dirname(ROOT);
export const SKILL = 'C:/Users/Shirshas Laptop/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
export const RUNTIME = 'C:/Users/Shirshas Laptop/.cache/codex-runtimes/codex-primary-runtime';
export const C = {paper:'#FAFAF7',ink:'#122C39',muted:'#53636C',teal:'#007F7B',blue:'#537CA3',orange:'#D87A3F',light:'#DFE4E3',white:'#FFFFFF',red:'#B85C56'};
export const FONT = 'Arial';
export const p = Presentation.create({slideSize:{width:1280,height:720}});

export function text(s, str, x, y, w, h, opt={}) {
  const t=s.shapes.add({name:opt.name??'text',geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  t.text=str;
  t.text.style={typeface:FONT,fontSize:opt.size??24,color:opt.color??C.ink,bold:opt.bold??false,alignment:opt.align??'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0,...opt.style};
  return t;
}

export function slide(title, opt={}) {
  const s=p.slides.add();s.background.fill=opt.dark?C.ink:C.paper;
  const ink=opt.dark?C.white:C.ink;
  if(title) text(s,title,64,52,1152,105,{size:43,bold:true,color:ink});
  if(opt.subtitle) text(s,opt.subtitle,64,153,1148,52,{size:22,color:opt.dark?'#C8D7DA':C.muted});
  const n=p.slides.items.length;
  text(s,`${String(n).padStart(2,'0')}`,1170,670,46,24,{size:16,color:opt.dark?'#C8D7DA':C.muted,align:'right'});
  if(opt.note) s.speakerNotes.text=opt.note;
  return s;
}

export function note(s, body, sources=[]) {
  s.speakerNotes.text=body+(sources.length?'\n\nSources\n'+sources.join('\n'):'');
}

export async function image(s, filename, x,y,w,h,alt='Experiment evidence') {
  const file=path.isAbsolute(filename)?filename:path.join(PROJECT,filename);
  const b=await fs.readFile(file);
  return s.images.add({name:alt,blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:/\.jpe?g$/i.test(file)?'image/jpeg':'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}

export function table(s, values, x,y,w,h,widths,opts={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.styleOptions={headerRow:false,bandedRows:false};
  t.borders.assign({style:'solid',fill:C.light,width:0.6});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.paper,textStyle:{typeface:FONT,fontSize:opts.size??23,color:C.ink},margins:{top:14,bottom:12,left:12,right:12},anchor:'center'});
  t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.ink,textStyle:{typeface:FONT,fontSize:opts.size??23,bold:true,color:C.white}});
  if(opts.highlightRows) for(const r of opts.highlightRows)t.cells.block({row:r,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#E7F1ED',textStyle:{typeface:FONT,fontSize:opts.size??23,color:C.ink,bold:true}});
  return t;
}

export async function chart(s, categories, series,x,y,w,h,opts={}) {
  const {applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
  const colors=[C.muted,C.teal,C.orange,C.blue,C.red];
  const ch=s.charts.add(opts.type??'bar',{
    position:{left:x,top:y,width:w,height:h},
    categories,
    // Excel workbooks support 15 significant digits. Eight decimal places
    // preserve far more precision than the displayed three-decimal labels.
    series:series.map((a,i)=>({...a,values:a.values.map(v=>v===null?null:Number(v.toFixed(8))),fill:a.fill??colors[i],line:{style:'solid',fill:a.fill??colors[i],width:3},marker:{symbol:'circle',size:6,fill:a.fill??colors[i]}})),
    hasLegend:series.length>1,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:C.ink}},
    barOptions:{direction:opts.horizontal?'bar':'column',grouping:'clustered',gapWidth:80},
    lineOptions:{smooth:false},
    xAxis:{textStyle:{fontSize:opts.axisSize??19,fill:C.ink},line:{fill:C.light,width:1},majorGridlines:null},
    yAxis:{title:opts.yTitle?{text:opts.yTitle,textStyle:{fontSize:20,fill:C.ink}}:undefined,min:opts.min??0,max:opts.max,numberFormatCode:opts.format??'0.000',textStyle:{fontSize:19,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.light,width:0.7}},
    dataLabels:{showValue:opts.labels??false,position:'outEnd',numberFormatCode:opts.format??'0.000',textStyle:{fontSize:19,fill:C.ink}},
    chartFill:C.paper,chartLine:{fill:'none',width:0},plotAreaFill:C.paper,plotAreaLine:{fill:'none',width:0},
  });
  applyPresentationChartFont(ch,{fontFamily:FONT});
  return ch;
}

export async function finalize(name,requirements={},deliveryName) {
  const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
  const stage=path.join(ROOT,'.build');await fs.mkdir(stage,{recursive:true});
  const out=path.join(PROJECT,'output');await fs.mkdir(out,{recursive:true});
  const validatedDir=deliveryName?path.join(stage,'validated'):out;
  await fs.mkdir(validatedDir,{recursive:true});
  const candidatePath=path.join(stage,`${name}.candidate.pptx`);
  await(await PresentationFile.exportPptx(p)).save(candidatePath);
  const result=await finalizePresentation({
    workspaceDir:PROJECT,candidatePath,finalPath:path.join(validatedDir,`${name}.pptx`),
    pythonExecutable:path.join(RUNTIME,'dependencies/python/python.exe'),
    integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...(requirements.requiredNativeTableOwnerSlides??[]).flatMap(n=>['--require-native-table-slide',String(n)])],
    fontPolicy:{basis:'design',families:[FONT]},
    verifyArtifactToolImport:true,
    materializeLiteralChartWorkbooks:true,
    receiptPath:path.join(stage,`${name}.validation.json`),
    ...requirements,
  });
  console.log(JSON.stringify(result));
  await fs.mkdir(path.join(stage,'renders'),{recursive:true});
  for(const [i,s] of p.slides.items.entries()) {
    const png=await p.export({slide:s,format:'png',scale:1.5});
    await fs.writeFile(path.join(stage,'renders',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  const montage=await p.export({format:'webp',montage:true});
  await fs.writeFile(path.join(stage,`${name}.montage.webp`),new Uint8Array(await montage.arrayBuffer()));
  let deliveredPath;
  if(deliveryName){
    deliveredPath=path.join(out,deliveryName);
    const previous=await fs.stat(deliveredPath).catch(()=>null);
    if(previous){
      const revisions=path.join(stage,'previous-deliveries');
      await fs.mkdir(revisions,{recursive:true});
      await fs.copyFile(deliveredPath,path.join(revisions,`${Date.now()}-${deliveryName}`));
    }
    await fs.copyFile(result.finalPath,deliveredPath);
  }
  return {...result,...(deliveredPath?{deliveredPath}:{})};
}
