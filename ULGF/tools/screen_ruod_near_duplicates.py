"""CPU-only cross-split pHash/dHash screening. Never edits split membership."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .prepare_ruod_split import partition, read
    from .audit_ruod_dataset import write_json, file_hash
except ImportError:
    from prepare_ruod_split import partition, read
    from audit_ruod_dataset import write_json, file_hash

RESAMPLE = getattr(Image, 'Resampling', Image).LANCZOS
GRID = np.arange(32)
DCT = np.cos(np.pi * (2 * GRID[None, :] + 1) * GRID[:, None] / 64)
DCT[0] *= 1 / np.sqrt(2)
DCT *= np.sqrt(2 / 32)
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hashes(image):
    gray = image.convert('L')
    pixels = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.float64)
    low = (DCT @ pixels @ DCT.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False  # Ignore DC (average luminance).
    small = np.asarray(gray.resize((9, 8), RESAMPLE), dtype=np.int16)
    return np.packbits(bits).tobytes().hex(), np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()


def screen_pairs(left, right, phash_limit=8, dhash_limit=6, keep=200):
    """All cross pairs tested; retain bounded strongest candidates, not decisions."""
    rp = np.array([list(bytes.fromhex(r['phash'])) for r in right], dtype=np.uint8)
    rd = np.array([list(bytes.fromhex(r['dhash'])) for r in right], dtype=np.uint8)
    candidates, total = [], 0
    for row in left:
        lp = np.array(list(bytes.fromhex(row['phash'])), dtype=np.uint8)
        ld = np.array(list(bytes.fromhex(row['dhash'])), dtype=np.uint8)
        pd = POPCOUNT[rp ^ lp].sum(axis=1)
        dd = POPCOUNT[rd ^ ld].sum(axis=1)
        indices = np.flatnonzero((pd <= phash_limit) | (dd <= dhash_limit))
        total += len(indices)
        for j in indices:
            candidates.append(dict(left=row, right=right[int(j)], phash_distance=int(pd[j]),
                                   dhash_distance=int(dd[j]), score=int(pd[j] + dd[j])))
        if len(candidates) > keep * 4:
            candidates = sorted(candidates, key=rank)[:keep]
    return sorted(candidates, key=rank)[:keep], total


def rank(candidate):
    return (candidate['score'], candidate['phash_distance'], candidate['dhash_distance'],
            candidate['left']['id'], candidate['right']['id'])


def inventory(source, audit_root, split_root=None):
    """Use supplied split manifests, or reproduce the already approved seed-42 plan."""
    datasets, records = {}, {}
    summary = read(audit_root / 'summary.json')
    for split in ('train', 'test'):
        ann = source / 'RUOD_ANN' / ('instances_' + split + '.json')
        if file_hash(ann) != summary['splits'][split]['annotation_sha256']:
            raise ValueError('Original annotations changed')
        data = read(ann)
        records[split] = read(audit_root / 'audit' / (split + '.json'))
        lookup = {r['id']: r for r in records[split]}
        data['images'] = [dict(i, file_name=str(i['id'])+'.jpg',
                              width=lookup[i['id']]['stored_size'][0],
                              height=lookup[i['id']]['stored_size'][1]) for i in data['images']]
        datasets[split] = data
    if split_root:
        selected = [read(split_root / 'annotations' / ('instances_' + s + '.json'))
                    for s in ('train', 'validation', 'test')]
    else:
        selected = partition(datasets['train'], datasets['test'], records)[:3]
    rows = []
    for split, data in zip(('train', 'validation', 'test'), selected):
        origin = 'test' if split == 'test' else 'train'
        lookup = {r['id']: r for r in records[origin]}
        for image in data['images']:
            record = lookup[image['id']]
            rows.append(dict(split=split, origin=origin, id=image['id'], file_name=record['file_name'],
                             source_sha256=record['file_sha256'],
                             path=str(source / 'RUOD_pic' / origin / record['file_name'])))
    return rows


def fingerprint(args):
    row, cached = args
    blob = Path(row['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['source_sha256']:
        raise ValueError('Source differs from completed audit: ' + row['path'])
    key = row['source_sha256']
    if key in cached:
        return dict(row, **cached[key])
    with Image.open(io.BytesIO(blob)) as image:
        size = image.size
        image.draft('RGB', (256, 256))  # CPU JPEG reduced-resolution decode.
        image.load()
        phash, dhash = hashes(image)  # Stored pixels, consistent with derived dataset.
    return dict(row, phash=phash, dhash=dhash, width=size[0], height=size[1])


def contact_sheet(candidates, path, start=0, count=8):
    candidates = candidates[start:start+count]
    if not candidates:
        return
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        try:
            font = ImageFont.truetype('arial.ttf', 17)
        except OSError:
            font = ImageFont.load_default()
    canvas = Image.new('RGB', (1200, 300*len(candidates)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, candidate in enumerate(candidates):
        y = index*300
        draw.text((10, y+2), '#{} | pHash {} /63 | dHash {} /64 | candidate, NOT a duplicate verdict'.format(
            start+index+1, candidate['phash_distance'], candidate['dhash_distance']), fill='black', font=font)
        for col, side in enumerate(('left', 'right')):
            row = candidate[side]
            draw.text((col*600+10, y+24), '{} | {} | id {}'.format(row['split'], row['file_name'], row['id']),
                      fill='black', font=font)
            with Image.open(row['path']) as source:
                image = source.convert('RGB')
            image.thumbnail((580, 245), RESAMPLE)
            canvas.paste(image, (col*600+(600-image.width)//2, y+50+(245-image.height)//2))
    canvas.save(path)


def run(source, audit_root, output, split_root=None):
    output.mkdir(parents=True, exist_ok=True)
    rows = inventory(source, audit_root, split_root)
    cache_path = output / 'fingerprints.json'
    cached = {}
    if cache_path.exists():
        for r in read(cache_path):
            cached[r['source_sha256']] = {k: r[k] for k in ('phash', 'dhash', 'width', 'height')}
    fingerprints = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(rows), 64):
            fingerprints.extend(executor.map(fingerprint, [(r, cached) for r in rows[start:start+64]]))
            if start % 512 == 0:
                print('Fingerprinted {}/{}'.format(len(fingerprints), len(rows)), flush=True)
                write_json(cache_path, fingerprints)
    write_json(cache_path, fingerprints)
    report = dict(method='pHash63 + dHash64, stored raster, CPU JPEG draft decode',
                  thresholds=dict(phash_max=8, dhash_max=6, rule='OR'),
                  membership_changed=False, review_status='VISUAL_REVIEW_REQUIRED',
                  limitation='Heuristic screening; crops, mirrors and subtle variants can be missed.',
                  membership_source='provided split manifests' if split_root else 'reproduced seed-42 split from local audit',
                  splits={s: sum(r['split']==s for r in rows) for s in ('train', 'validation', 'test')}, boundaries={})
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        left = [r for r in fingerprints if r['split']==a]
        right = [r for r in fingerprints if r['split']==b]
        candidates, total = screen_pairs(left, right)
        name = a + '__' + b
        write_json(output / (name + '.json'), candidates)
        contact_sheet(candidates, output / (name + '.png'))
        report['boundaries'][name] = dict(pairs_screened=len(left)*len(right),
                                         threshold_candidates=total, saved_strongest=len(candidates),
                                         contact_sheet_pairs=min(8, len(candidates)))
        print(name + ': ' + json.dumps(report['boundaries'][name]), flush=True)
    write_json(output / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--splits', type=Path)
    args = parser.parse_args()
    run(args.source, args.audit, args.output, args.splits)
