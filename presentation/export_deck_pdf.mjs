/** Portable PDF copy of the inspected slide renders. Editable source is PPTX. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { PDFDocument } from 'pdf-lib';
import { fileURLToPath } from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const renders=path.join(here,'.build/renders');
const files=(await fs.readdir(renders)).filter(n=>/^slide-\d\d\.png$/.test(n)).sort();
if(files.length!==12)throw new Error(`Expected12renderedslides,found${files.length}`);
const pdf=await PDFDocument.create();
pdf.setTitle('Color-Constancy-Assisted Monocular Depth Estimation: Thesis Progress');
pdf.setAuthor('Shirsha Mitra,2023521460130');
pdf.setSubject('Measured first controlled pilot experiment');
for(const file of files){
  const img=await pdf.embedPng(await fs.readFile(path.join(renders,file)));
  const page=pdf.addPage([960,540]);
  page.drawImage(img,{x:0,y:0,width:960,height:540});
}
const out=path.join(here,'../output/Shirsha_Mitra_Thesis_Progress_Slides.pdf');
await fs.writeFile(out,await pdf.save());
console.log(out);
