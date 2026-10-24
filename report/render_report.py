"""Render every final PDF page for visual quality assurance with PyMuPDF."""
import argparse
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf', type=Path)
    ap.add_argument('--outdir', type=Path, default=Path(__file__).parent/'tmp'/'pdfs')
    ap.add_argument('--dpi', type=int, default=120)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    meta = []
    thumbs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        name = args.outdir/f'page-{i+1:02d}.png'
        pix.save(name)
        blocks = page.get_text('blocks')
        out_of_bounds = [list(b[:4]) for b in blocks
                         if b[0] < -0.1 or b[1] < -0.1 or b[2] > page.rect.width+0.1
                         or b[3] > page.rect.height+0.1]
        meta.append({'page': i+1, 'text_characters': len(page.get_text()),
                     'raster': str(name), 'out_of_bounds_blocks': out_of_bounds})
        im = Image.open(name).convert('RGB')
        im.thumbnail((350, 495), Image.Resampling.LANCZOS)
        thumbs.append(im)
    cols = 4
    rows = (len(thumbs)+cols-1)//cols
    sheet = Image.new('RGB', (cols*370, rows*530), '#dce1e5')
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs):
        x, y = (i % cols)*370+10, (i//cols)*530+10
        sheet.paste(im, (x, y))
        d.text((x, y+499), f'Page {i+1}', fill='#13263b')
    sheet.save(args.outdir/'contact-sheet.png')
    (args.outdir/'qa.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'pages': len(doc), 'render_directory': str(args.outdir),
                      'out_of_bounds': sum(len(x['out_of_bounds_blocks']) for x in meta)},indent=2))


if __name__ == '__main__':
    main()
