"""Build Shirsha Mitra's evidence-backed thesis progress report.

Usage: python report/build_report.py --payload report/report_payload.json
All experimental claims and values come from the supplied payload. No demo
numbers are inserted, and missing required fields stop report creation.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
W, H = 595.276, 841.89
M = 43
CW = W - 2 * M
NAVY = colors.HexColor('#13263B')
INK = colors.HexColor('#23394C')
TEAL = colors.HexColor('#007C80')
MINT = colors.HexColor('#E8F4F1')
ORANGE = colors.HexColor('#D87D35')
LIGHT = colors.HexColor('#F0F3F5')
GREY = colors.HexColor('#657788')
LINE = colors.HexColor('#D5DEE4')
WHITE = colors.white


def register_fonts():
    fontdir = Path('C:/Windows/Fonts')
    choices = [('Body', 'arial.ttf'), ('Bold', 'arialbd.ttf'), ('Italic', 'ariali.ttf')]
    for name, filename in choices:
        p = fontdir / filename
        if not p.exists():
            raise FileNotFoundError(f'Required font missing: {p}')
        pdfmetrics.registerFont(TTFont(name, str(p)))


class Report:
    def __init__(self, payload: dict, output: Path):
        self.p = payload
        self.output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1)
        self.c.setTitle('Color Constancy and Monocular Depth - Research Progress')
        self.c.setAuthor('Shirsha Mitra')
        self.page_n = 0
        self.overflow = []

    def rect(self, x, y, w, h, fill, radius=0, stroke=None):
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke or fill)
        if radius:
            self.c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=bool(stroke))
        else:
            self.c.rect(x, H-y-h, w, h, fill=1, stroke=bool(stroke))

    def line(self, x, y, x2, y2, color=LINE, width=0.6):
        self.c.setStrokeColor(color)
        self.c.setLineWidth(width)
        self.c.line(x, H-y, x2, H-y2)

    def text(self, text, x, y, size=10, color=INK, font='Body'):
        self.c.setFillColor(color)
        self.c.setFont(font, size)
        self.c.drawString(x, H-y-size, str(text))

    def para(self, text, x, y, w, size=10, color=INK, leading=None, bold=False,
             max_h=None, raw=False):
        text = str(text)
        if not raw:
            text = escape(text).replace('\n', '<br/>')
        st = ParagraphStyle('body', fontName='Bold' if bold else 'Body', fontSize=size,
                            leading=leading or size*1.4, textColor=color,
                            spaceAfter=0, alignment=TA_LEFT)
        p = Paragraph(text, st)
        _, h = p.wrap(w, H)
        if max_h is not None and h > max_h+0.2:
            self.overflow.append(f'page {self.page_n}: text at ({x},{y}) height {h:.1f}>{max_h}: {text[:90]}')
        if y+h > H-44:
            self.overflow.append(f'page {self.page_n}: text below footer at {y+h:.1f}: {text[:90]}')
        p.drawOn(self.c, x, H-y-h)
        return y+h

    def img(self, path, x, y, w, h, bg=LIGHT, crop=False):
        p = Path(path)
        if not p.is_absolute():
            p = ROOT/p
        if not p.exists():
            raise FileNotFoundError(p)
        self.rect(x, y, w, h, bg, 5)
        with Image.open(p) as im:
            iw, ih = im.size
        s = max(w/iw, h/ih) if crop else min(w/iw, h/ih)
        dw, dh = iw*s, ih*s
        self.c.saveState()
        clip = self.c.beginPath()
        clip.roundRect(x, H-y-h, w, h, 5)
        self.c.clipPath(clip, stroke=0, fill=0)
        self.c.drawImage(str(p), x+(w-dw)/2, H-y-h+(h-dh)/2,
                         dw, dh, preserveAspectRatio=True, mask='auto')
        self.c.restoreState()

    def page(self, section=None, title=None, subtitle=None):
        if self.page_n:
            self.c.showPage()
        self.page_n += 1
        self.rect(0, 0, W, H, WHITE)
        self.rect(0, 0, W, 6, TEAL)
        self.text('SHIRSHA MITRA  /  THESIS RESEARCH', M, 23, 8, GREY, 'Bold')
        self.line(M, H-41, W-M, H-41)
        self.text('COLOR CONSTANCY + MONOCULAR DEPTH', M, H-30, 7, GREY)
        self.c.setFillColor(GREY)
        self.c.setFont('Body', 8)
        self.c.drawRightString(W-M, 20, f'{self.page_n:02d} / 08')
        if section:
            self.text(section.upper(), M, 63, 8.2, TEAL, 'Bold')
        if title:
            self.para(title, M, 85, CW, 24, NAVY, leading=28, bold=True, max_h=60)
        if subtitle:
            self.para(subtitle, M, 146, CW, 10.5, GREY, leading=14, max_h=45)

    def card(self, x, y, w, h, title, body, accent=TEAL):
        self.rect(x, y, w, h, LIGHT, 7)
        self.rect(x, y+12, 3, 20, accent, 1)
        self.para(title, x+15, y+12, w-28, 11.5, NAVY, bold=True, max_h=36)
        self.para(body, x+15, y+48, w-28, 9.5, INK, max_h=h-59)

    def bullets(self, items, x, y, w, size=10, gap=12):
        for item in items:
            self.rect(x, y+6, 4, 4, TEAL, 2)
            y = self.para(item, x+15, y, w-15, size, leading=size*1.4)+gap
        return y

    def table(self, headers, rows, widths, x, y, row_h=35, size=9.3,
              highlight=None):
        self.rect(x, y, sum(widths), 30, NAVY, 3)
        xx=x
        for h,w in zip(headers,widths):
            self.para(h,xx+9,y+7,w-17,8.8,WHITE,bold=True,max_h=23)
            xx+=w
        y+=30
        for i,row in enumerate(rows):
            fill=MINT if highlight is not None and i==highlight else (LIGHT if i%2==0 else WHITE)
            self.rect(x,y,sum(widths),row_h,fill)
            xx=x
            for value,w in zip(row,widths):
                self.para(value,xx+9,y+9,w-17,size,leading=size*1.25,max_h=row_h-15)
                xx+=w
            self.line(x,y+row_h,x+sum(widths),y+row_h)
            y+=row_h
        return y

    def figure(self, fig, x, y, w, h, caption_y=None):
        self.img(fig['path'],x,y,w,h)
        if fig.get('caption'):
            self.para(fig['caption'],x,caption_y or y+h+9,w,8.5,GREY,max_h=45)

    def cover(self):
        p=self.p
        self.page()
        self.text('RESEARCH PROGRESS REPORT',M,68,9,TEAL,'Bold')
        self.para('Color Constancy\nMeets Monocular Depth',M,100,CW,32,NAVY,leading=37,bold=True,max_h=84)
        self.para(p['thesis_title'],M,194,CW,11.2,GREY,leading=16,max_h=48)
        self.text('Shirsha Mitra',M,260,15,NAVY,'Bold')
        self.text('Student ID 2023521460130',M,283,10,GREY)
        self.text(p['date_label'],M,304,10,GREY)
        self.text(p.get('phase_label','INITIAL CONTROLLED EXPERIMENT'),M,336,8.2,TEAL,'Bold')
        self.figure(p['cover_figure'],M,358,CW,237)
        stats=p['cover_stats']
        gap=10; sw=(CW-2*gap)/3
        for i,s in enumerate(stats):
            x=M+i*(sw+gap)
            self.rect(x,645,sw,65,LIGHT,6)
            self.text(s['value'],x+13,654,22,TEAL,'Bold')
            self.para(s['label'],x+13,682,sw-23,8.4,GREY,max_h=23)
        self.para(p['executive_takeaway'],M,730,CW,10.1,NAVY,bold=True,max_h=43)

    def question(self):
        p=self.p
        self.page('01 / Research question','Can color correction make depth estimates more reliable?',
                  'The same scene keeps its geometry when its image colors change.')
        self.rect(M,204,CW,94,MINT,8)
        self.para(p['hypothesis'],M+17,221,CW-34,13,NAVY,leading=19,bold=True,max_h=60)
        y=324; cw=(CW-24)/3
        stages=[('01  RGB image','One ordinary color photograph.'),
                ('02  Color correction','Estimate and reduce a global color cast.'),
                ('03  Depth prediction','Predict the distance of visible scene points.')]
        for i,(a,b) in enumerate(stages):
            self.card(M+i*(cw+12),y,cw,127,a,b)
        self.text('WHAT THIS PILOT IMPLEMENTS',M,483,9,TEAL,'Bold')
        self.bullets(p['implemented_methods'],M,509,CW,10.2,15)
        self.rect(M,681,CW,79,LIGHT,7)
        self.para('How to read the output',M+15,695,CW-30,11,NAVY,bold=True)
        self.para(p['depth_explanation'],M+15,718,CW-30,9.6,INK,max_h=40)

    def protocol(self):
        p=self.p
        self.page('02 / Experimental design','A controlled, paired comparison',
                  'Every variant uses the same images, depth model, and evaluation pixels.')
        y=self.table(['Protocol item','Actual setting'],p['protocol_rows'],[137,CW-137],M,201,row_h=43,size=9.3)
        self.text('DEPTH METRICS',M,y+26,9,TEAL,'Bold')
        self.card(M,y+49,(CW-13)/2,103,'AbsRel  |  lower is better',
                  'Mean absolute depth error divided by the reference depth. A value of 0.10 corresponds to 10% relative error.')
        self.card(M+(CW+13)/2,y+49,(CW-13)/2,103,'RMSE  |  lower is better',
                  'Root mean squared distance error in meters. Larger mistakes receive more weight.')
        self.rect(M,699,CW,61,MINT,6)
        self.para(p['protocol_caution'],M+14,711,CW-28,9.6,NAVY,max_h=41)

    def results(self):
        p=self.p
        self.page('03 / Measured results',p['results_title'],p['results_subtitle'])
        end=self.table(p['results_headers'],p['results_rows'],p['results_widths'],M,205,row_h=p.get('results_row_height',34),size=9.2,highlight=p.get('highlight_row'))
        fig_y=end+27
        self.figure(p['results_figure'],M,fig_y,CW,218)
        self.rect(M,697,CW,64,MINT,7)
        self.para(p['results_takeaway'],M+15,709,CW-30,10.2,NAVY,bold=True,max_h=43)

    def stress(self):
        p=self.p
        self.page('04 / Robustness and variation',p['stress_title'],p['stress_subtitle'])
        self.figure(p['stress_figure'],M,206,CW,279)
        self.text('WHAT THE DISTRIBUTION SHOWS',M,538,9,TEAL,'Bold')
        self.bullets(p['stress_findings'],M,565,CW,10.2,16)
        if p.get('statistics_reading_note'):
            self.rect(M,735,CW,51,LIGHT,6)
            self.para(p['statistics_reading_note'],M+13,747,CW-26,8.4,GREY,max_h=30)

    def qualitative(self):
        p=self.p
        self.page('05 / Visual evidence','Visual evidence from a selected extreme',
                  p['qualitative_subtitle'])
        self.figure(p['qualitative_figure'],M,201,CW,267)
        stats=p.get('qualitative_stats',[])
        gap=10; sw=(CW-2*gap)/3
        for i,s in enumerate(stats):
            x=M+i*(sw+gap)
            self.rect(x,536,sw,110,LIGHT,6)
            self.para(s['title'],x+12,549,sw-24,9.1,NAVY,bold=True,max_h=27)
            self.text(s['value'],x+12,573,22,TEAL,'Bold')
            self.para(s['label'],x+12,605,sw-24,8.1,GREY,max_h=27)
        self.rect(M,689,CW,75,LIGHT,7)
        self.para(p['qualitative_note'],M+15,702,CW-30,9.6,INK,max_h=52)

    def interpretation(self):
        p=self.p
        self.page('06 / Interpretation','What this experiment establishes',
                  'A first reproducible result, followed by a focused research plan.')
        y=205
        for i,item in enumerate(p['findings']):
            self.text(f'{i+1:02d}',M,y,20,TEAL,'Bold')
            self.para(item['title'],M+42,y+1,CW-42,11.5,NAVY,bold=True,max_h=30)
            self.para(item['body'],M+42,y+29,CW-42,9.8,INK,max_h=55)
            y+=100
        self.text('LIMITS OF THE CURRENT EVIDENCE',M,531,9,ORANGE,'Bold')
        self.bullets(p['limitations'],M,556,CW,9.4,10)
        self.rect(M,708,CW,56,MINT,6)
        self.para(p['next_step'],M+14,719,CW-28,9.7,NAVY,bold=True,max_h=39)

    def reproducibility(self):
        p=self.p
        self.page('07 / Reproducibility','From laptop setup to saved evidence',
                  'Code, configuration, and numerical outputs support every claim in this report.')
        self.para(p['environment_summary'],M,204,CW,9.6,INK,max_h=58)
        self.text('RE-RUN FROM THE PROJECT DIRECTORY',M,276,8.8,TEAL,'Bold')
        y=300
        for cmd in p['commands']:
            lines=textwrap.wrap(cmd,87,break_long_words=True,break_on_hyphens=False)
            h=19+len(lines)*11
            self.rect(M,y,CW,h,LIGHT,4)
            self.para('\n'.join(lines),M+11,y+9,CW-22,8.2,INK,leading=11,max_h=h-15)
            y+=h+8
        self.para(p['reproducibility_note'],M,y+7,CW,8.8,GREY,max_h=49)
        self.text('SOURCES AND PROJECT REFERENCES',M,556,8.8,TEAL,'Bold')
        y=580
        for i,s in enumerate(p['sources'],1):
            self.para(f'{i}. {s["title"]}',M,y,CW,8.8,NAVY,bold=True,max_h=25)
            y+=15
            self.para(s['url'],M,y,CW,7.7,TEAL,max_h=24)
            # The source title is clickable while the full address remains printable.
            self.c.linkURL(s['url'],(M,H-y-24,W-M,H-y+16),relative=0,thickness=0)
            y+=31
        self.rect(M,732,CW,34,MINT,5)
        self.para('Prepared for research supervision. All numerical findings describe this pilot run only.',M+12,742,CW-24,8.1,NAVY,max_h=15)

    def build(self):
        for fn in [self.cover,self.question,self.protocol,self.results,self.stress,
                   self.qualitative,self.interpretation,self.reproducibility]:
            fn()
        self.c.save()
        if self.overflow:
            raise RuntimeError('Layout bounds failed:\n'+'\n'.join(self.overflow))
        return self.output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--payload',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'output'/'pdf'/'Shirsha_Mitra_Thesis_Progress_Report.pdf')
    args=ap.parse_args()
    payload=json.loads(args.payload.read_text(encoding='utf-8'))
    if not payload.get('evidence_verified'):
        raise ValueError('Report payload must set evidence_verified=true after checking real experiment outputs.')
    register_fonts()
    out=Report(payload,args.output).build()
    print(out)


if __name__=='__main__':
    main()
