"""Render stored-coordinate boxes for quarantined EXIF-0 images; no source edits."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tools.audit_ruod_dataset import file_hash, valid_box


def render(source, audit_root, output):
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except OSError:
            font = ImageFont.load_default()
    panels, records = [], []
    for split in ('train', 'test'):
        data = json.loads((source / 'RUOD_ANN' / ('instances_' + split + '.json')).read_text())
        images = {i['id']: i for i in data['images']}
        categories = {c['id']: c['name'] for c in data['categories']}
        audit = json.loads((audit_root / 'audit' / (split + '.json')).read_text())
        for row in audit:
            if row.get('exif_orientation') != 0:
                continue
            item = images[row['id']]
            path = source / 'RUOD_pic' / split / item['file_name']
            digest = file_hash(path)
            if digest != row['file_sha256']:
                raise ValueError('Source changed since audit: ' + str(path))
            with Image.open(path) as image:
                image.load()
                assert image.getexif().get(274) == 0
                rgb = image.convert('RGB')  # Deliberately no EXIF transpose.
            anns = [a for a in data['annotations'] if a['image_id'] == item['id']]
            assert rgb.size == (item['width'], item['height'])
            assert all(valid_box(a['bbox'], rgb.size) for a in anns)
            preview = rgb.copy()
            preview.thumbnail((600, 420))
            panel = Image.new('RGB', (640, 500), 'white')
            ox, oy = (640 - preview.width) // 2, 65 + (420 - preview.height) // 2
            panel.paste(preview, (ox, oy))
            draw = ImageDraw.Draw(panel)
            draw.text((10, 8), '{} | {} | id {}'.format(split, item['file_name'], item['id']), fill='black', font=font)
            draw.text((10, 30), '{}x{} | EXIF 0 | {} boxes | RAW pixels'.format(*rgb.size, len(anns)), fill='black', font=font)
            for ann in anns:
                x, y, w, h = ann['bbox']
                sx, sy = preview.width / rgb.width, preview.height / rgb.height
                box = (ox+x*sx, oy+y*sy, ox+(x+w)*sx, oy+(y+h)*sy)
                draw.rectangle(box, outline='#ff3b30', width=2)
                label = categories[ann['category_id']]
                tx, ty = int(box[0]), max(65, int(box[1])-19)
                draw.rectangle((tx, ty, min(639, tx+len(label)*10+4), ty+19), fill='black')
                draw.text((tx+2, ty), label, fill='yellow', font=font)
            panel.save(output / ('{}_{}.png'.format(split, item['id'])))
            panels.append(panel)
            records.append(dict(split=split, id=item['id'], file_name=item['file_name'],
                                sha256=digest, boxes=len(anns), stored_size=list(rgb.size)))
    sheet = Image.new('RGB', (1280, ((len(panels)+1)//2)*500), '#cccccc')
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2)*640, (i // 2)*500))
    sheet.save(output / 'contact_sheet.png')
    (output / 'review_inventory.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.source, args.audit, args.output)
