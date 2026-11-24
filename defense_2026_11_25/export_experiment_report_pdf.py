"""Searchable PDF with the same paragraphs, tables and figures as the DOCX."""
from pathlib import Path
from html import escape
import hashlib
import json
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph as WordParagraph
from docx.table import Table as WordTable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle, Image, Spacer, PageBreak

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
OUT = ROOT / 'output/defense_2026_11_25'
STEM = 'Shirsha_Mitra_Proposal_Report_With_Experiments'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = OUT / f'{STEM}.docx'
    target = OUT / f'{STEM}.pdf'
    doc = Document(source)
    for name, file in [('ReportTimes', 'times.ttf'), ('ReportTimesBold', 'timesbd.ttf'),
                       ('ReportHeading', 'timesbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
    pdfmetrics.registerFontFamily('ReportTimes', normal='ReportTimes', bold='ReportTimesBold', italic='ReportTimes', boldItalic='ReportTimesBold')
    sec = doc.sections[0]
    width, height = sec.page_width.pt, sec.page_height.pt
    left, right, top, bottom = [getattr(sec, key).pt for key in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin')]
    story = []
    names = ['input_clouds', 'training_curves', 'noise_results', 'illuminant_outputs']
    figure_count = 0
    for child in doc.element.body:
        if child.tag == qn('w:p'):
            wp = WordParagraph(child, doc)
            if child.xpath('.//w:br[@w:type="page"]'):
                story.append(PageBreak())
                continue
            drawings = child.xpath('.//wp:inline')
            if drawings:
                figure_name = names[figure_count]
                extent = drawings[0].find(qn('wp:extent'))
                pic = Image(str(HOME / f'.build/experiment_report_figures/{figure_name}.png'),
                            width=int(extent.get('cx')) / 12700, height=int(extent.get('cy')) / 12700)
                pic.hAlign = 'CENTER'
                pic.spaceAfter = 2
                pic.keepWithNext = True
                story.append(pic)
                figure_count += 1
                continue
            if not wp.text:
                story.append(Spacer(1, 8))
                continue
            name = wp.style.name
            size = next((r.font.size.pt for r in wp.runs if r.font.size is not None), None)
            is_heading = name.startswith('Heading') or name == 'Title'
            defaults = {'Title': 18, 'Heading 1': 13, 'Heading 2': 11.5}
            size = size or defaults.get(name, 11)
            pf = wp.paragraph_format
            after = pf.space_after.pt if pf.space_after is not None else (4 if is_heading else 5)
            before = pf.space_before.pt if pf.space_before is not None else (8 if is_heading else 0)
            body = []
            for run in wp.runs:
                text = escape(run.text).replace('\n', '<br/>')
                body.append(f'<b>{text}</b>' if run.bold and not is_heading else text)
            style = ParagraphStyle('content', fontName='ReportHeading' if is_heading else 'ReportTimes',
                                   fontSize=size, leading=size * (1.05 if size >= 10.5 else 1.05),
                                   textColor=colors.black, spaceBefore=before, spaceAfter=after,
                                   keepWithNext=is_heading, allowWidows=0, allowOrphans=0)
            story.append(Paragraph(''.join(body), style))
        elif child.tag == qn('w:tbl'):
            wt = WordTable(child, doc)
            rows = []
            for i, row in enumerate(wt.rows):
                cells = []
                for j, cell in enumerate(row.cells):
                    bold = i == 0 or any(r.bold for p in cell.paragraphs for r in p.runs)
                    cell_style = ParagraphStyle('cell', fontName='ReportTimesBold' if bold else 'ReportTimes',
                                                fontSize=10, leading=10.5, textColor=colors.black,
                                                alignment=TA_LEFT if j == 0 else TA_CENTER)
                    cells.append(Paragraph(escape(cell.text).replace('\n', '<br/>'), cell_style))
                rows.append(cells)
            widths = [col.width.pt for col in wt.columns]
            t = Table(rows, colWidths=widths, repeatRows=1, hAlign='CENTER')
            t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), .5, colors.black),
                                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                    ('LEFTPADDING', (0, 0), (-1, -1), 4.25),
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 4.25),
                                    ('TOPPADDING', (0, 0), (-1, -1), 3.75),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3.75)]))
            story.append(t)
    assert figure_count == 4

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.black)
        canvas.setFont('ReportTimes', 9)
        canvas.drawCentredString(width / 2, 22, f'Shirsha Mitra    {document.page}')
        canvas.restoreState()

    output = BaseDocTemplate(str(target), pagesize=(width, height), leftMargin=left, rightMargin=right,
                             topMargin=top, bottomMargin=bottom, title='Undergraduate Thesis Proposal with Experimental Results',
                             author='Shirsha Mitra', subject='Depth reliability aware fusion for point cloud color constancy',
                             pageCompression=1)
    frame = Frame(left, bottom, width - left - right, height - top - bottom,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    output.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=footer))
    output.build(story)
    receipt = {'status': 'PASS', 'renderer': 'ReportLab searchable PDF from DOCX content',
               'source_sha256': sha(source), 'pdf_sha256': sha(target), 'figures': figure_count,
               'pdf': str(target), 'pdf_exporter_sha256': sha(Path(__file__))}
    (HOME / 'experiment_report_pdf_export.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
