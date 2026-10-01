"""Incremental reviewed restoration and deterministic exact-duplicate-safe split.

Uses the completed audit. Decodes only the ten reviewed images, not the dataset.
No model, CUDA, or training dependencies. Originals are never modified.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import random
import shutil

from PIL import Image
try:
    from .audit_ruod_dataset import file_hash, pixel_hash, strip_jpeg_exif, valid_box, write_json
except ImportError:
    from audit_ruod_dataset import file_hash, pixel_hash, strip_jpeg_exif, valid_box, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def annotation_signature(image, annotations):
    """Exact agreement only; do not silently choose between conflicting labels."""
    objects = []
    for ann in annotations:
        objects.append(json.dumps({k: v for k, v in ann.items()
                                   if k not in ('id', 'image_id')}, sort_keys=True))
    return image['width'], image['height'], tuple(sorted(objects))


def partition(train, test, records, seed=42, validation_fraction=0.1):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    if train['categories'] != test['categories']:
        raise ValueError('Train/test category mapping differs')
    nodes = [(split, row['id']) for split in ('train', 'test') for row in records[split]]
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owners = {}
    for split in ('train', 'test'):
        for row in records[split]:
            node = (split, row['id'])
            for field in ('stored_pixels_sha256', 'display_pixels_sha256', 'file_sha256'):
                key = (field, row[field])
                if key in owners:
                    parent[find(node)] = find(owners[key])
                else:
                    owners[key] = node
    components = defaultdict(list)
    for node in nodes:
        components[find(node)].append(node)
    image_map = {row['id']: row for row in train['images']}
    anns = defaultdict(list)
    for ann in train['annotations']:
        anns[ann['image_id']].append(ann)
    retained, exclusions, duplicate_groups = [], [], []
    for component in components.values():
        ids = sorted(i for split, i in component if split == 'train' and i in image_map)
        if not ids:
            continue
        tests = sorted(i for split, i in component if split == 'test')
        if tests:
            exclusions.extend(dict(id=i, reason='exact_match_to_official_test', test_ids=tests) for i in ids)
        else:
            signatures = {annotation_signature(image_map[i], anns[i]) for i in ids}
            if len(signatures) != 1:
                exclusions.extend(dict(id=i, reason='conflicting_duplicate_annotations', group=ids) for i in ids)
            else:
                retained.append(ids[0])
                exclusions.extend(dict(id=i, reason='redundant_identical_annotation_copy', retained_id=ids[0])
                                  for i in ids[1:])
        if len(component) > 1:
            duplicate_groups.append([dict(split=s, id=i) for s, i in sorted(component)])
    retained.sort()
    random.Random(seed).shuffle(retained)
    count = int(round(len(retained) * validation_fraction))
    if count < 1 or count >= len(retained):
        raise ValueError('Too few eligible images for train/validation')
    val_ids, train_ids = set(retained[:count]), set(retained[count:])
    def make(ids, source):
        return dict(source, images=[dict(i, file_name='train/' + i['file_name'])
                                    for i in source['images'] if i['id'] in ids],
                    annotations=[a for a in source['annotations'] if a['image_id'] in ids])
    training, validation = make(train_ids, train), make(val_ids, train)
    official_test = dict(test, images=[dict(i, file_name='test/' + i['file_name']) for i in test['images']])
    # Independent hash-set check across every output partition.
    lookup = {(s, r['id']): r for s in records for r in records[s]}
    memberships = [set(('train', i) for i in train_ids), set(('train', i) for i in val_ids),
                   set(('test', i['id']) for i in test['images'])]
    for field in ('stored_pixels_sha256', 'display_pixels_sha256', 'file_sha256'):
        sets = [{lookup[node][field] for node in members} for members in memberships]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Leakage check failed: ' + field)
    distributions = {}
    for name, data in (('train', training), ('validation', validation), ('official_test', official_test)):
        counts = Counter(a['category_id'] for a in data['annotations'])
        if set(counts) != {c['id'] for c in data['categories']}:
            raise ValueError('A class is absent in ' + name)
        distributions[name] = {c['name']: counts[c['id']] for c in data['categories']}
    report = dict(seed=seed, validation_fraction=validation_fraction,
                  splitting_method='seeded image shuffle after exact deduplication; class coverage checked',
                  images={k: len(v['images']) for k, v in [('train', training), ('validation', validation),
                                                       ('official_test', official_test)]},
                  exclusion_counts=dict(Counter(e['reason'] for e in exclusions)),
                  annotations_per_class=distributions,
                  exact_hash_leakage='PASS', near_duplicate_checks='NOT_PERFORMED',
                  training_ready=False,
                  status='EXACT_DUPLICATE_SPLIT_PASS_NEAR_DUPLICATE_REVIEW_PENDING')
    return training, validation, official_test, exclusions, duplicate_groups, report


def prepare(source, derived, approvals, destination):
    source, derived, destination = source.resolve(), derived.resolve(), destination.resolve()
    if source == derived or source in derived.parents or source == destination or source in destination.parents:
        raise ValueError('Do not write into the source dataset')
    summary = read(derived / 'summary.json')
    policy = read(derived / 'policy.json')
    if Path(policy['source']).resolve() != source or not policy.get('derive'):
        raise ValueError('Wrong source or non-export audit policy')
    records, originals, datasets, targets = {}, {}, {}, []
    restored_annotations = Counter()
    for split in ('train', 'test'):
        ann_path = source / 'RUOD_ANN' / ('instances_' + split + '.json')
        if file_hash(ann_path) != summary['splits'][split]['annotation_sha256']:
            raise ValueError('Source annotations changed: ' + split)
        originals[split] = read(ann_path)
        datasets[split] = read(derived / 'annotations' / ('instances_' + split + '.json'))
        records[split] = read(derived / 'audit' / (split + '.json'))
        original_images = {i['id']: i for i in originals[split]['images']}
        original_anns = defaultdict(list)
        for ann in originals[split]['annotations']:
            original_anns[ann['image_id']].append(ann)
        for row in records[split]:
            if row.get('exif_orientation') != 0:
                if row['status'] != 'ACCEPTED':
                    raise ValueError('Unexpected unresolved quarantine: ' + str(row['id']))
                continue
            if set(row['issues']) - {'invalid_exif_orientation', 'coordinate_convention_needs_review'}:
                raise ValueError('Additional unreviewed defect')
            key = split + ':' + str(row['id'])
            path = source / 'RUOD_pic' / split / row['file_name']
            if approvals.get(key) != row['file_sha256'] or file_hash(path) != approvals[key]:
                raise ValueError('Missing approval or changed reviewed file: ' + str(path))
            item = original_images[row['id']]
            restored_bytes = strip_jpeg_exif(path.read_bytes())
            with Image.open(io.BytesIO(restored_bytes)) as image:
                image.load()
                if image.getexif() or pixel_hash(image) != row['stored_pixels_sha256']:
                    raise ValueError('Restored pixel/metadata check failed: ' + key)
                if image.size != (item['width'], item['height']):
                    raise ValueError('Unexpected dimensions: ' + key)
                if not all(valid_box(a['bbox'], image.size) for a in original_anns[row['id']]):
                    raise ValueError('Invalid box: ' + key)
            filename = str(row['id']) + '.jpg'
            target = derived / 'images' / split / filename
            if target.exists() and target.read_bytes() != restored_bytes:
                raise ValueError('Unexpected existing restored file: ' + str(target))
            targets.append((target, restored_bytes, key))
            restored_annotations[split] += len(original_anns[row['id']])
            # Idempotent metadata restoration; do not duplicate image/annotation IDs.
            datasets[split]['images'] = [i for i in datasets[split]['images'] if i['id'] != row['id']]
            datasets[split]['annotations'] = [a for a in datasets[split]['annotations'] if a['image_id'] != row['id']]
            datasets[split]['images'].append(dict(item, file_name=filename))
            datasets[split]['annotations'].extend(original_anns[row['id']])
            row.update(status='ACCEPTED', issues=[], raw_coordinate_approval=True,
                       derived_file='images/{}/{}'.format(split, filename),
                       resolution='reviewed_EXIF_0_removed_pixels_and_boxes_unchanged')
        datasets[split]['images'].sort(key=lambda i: i['id'])
        datasets[split]['annotations'].sort(key=lambda a: a['id'])
        data = datasets[split]
        if {i['id'] for i in data['images']} != set(original_images):
            raise ValueError('Restoration does not recover all original image IDs')
        if data['annotations'] != sorted(originals[split]['annotations'], key=lambda a: a['id']):
            raise ValueError('Annotation content differs from original')
        pending = {p for p, _, _ in targets}
        for item in data['images']:
            image_path = derived / 'images' / split / item['file_name']
            if image_path not in pending and not image_path.is_file():
                raise FileNotFoundError('Runtime export missing: ' + str(image_path))
    if len(targets) != 10:
        raise ValueError('Expected exactly ten reviewed EXIF-0 images')
    training, validation, test, exclusions, groups, report = partition(datasets['train'], datasets['test'], records)
    # Complete every preflight before changing the derived dataset; backup JSONs first.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = derived / 'restoration-backups' / stamp
    for name in ('policy.json', 'summary.json', 'audit/train.json', 'audit/test.json',
                 'annotations/instances_train.json', 'annotations/instances_test.json'):
        backup_path = backup / name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived / name, backup_path)
    for path, blob, key in targets:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.restore.tmp')
            temporary.write_bytes(blob)
            temporary.replace(path)
        print('RESTORED:', key, flush=True)
    for split in ('train', 'test'):
        data = datasets[split]
        write_json(derived / 'annotations' / ('instances_' + split + '.json'), data)
        write_json(derived / 'audit' / (split + '.json'), records[split])
        summary['splits'][split].update(accepted_images=len(data['images']),
            accepted_annotations=len(data['annotations']), quarantined_images=0,
            quarantined_annotations=0, issues={})
    policy['approved_raw_coordinates'] = approvals
    summary['quarantined_images'] = sum(s.get('quarantined_images', 0) for s in summary['splits'].values())
    summary['orientation_restoration'] = dict(status='PASS', reviewed_images=10, backup=str(backup))
    write_json(derived / 'policy.json', policy)
    write_json(derived / 'summary.json', summary)
    for split, data in (('train', training), ('validation', validation), ('test', test)):
        write_json(destination / 'annotations' / ('instances_' + split + '.json'), data)
    write_json(destination / 'exclusions.json', exclusions)
    write_json(destination / 'duplicate_groups.json', groups)
    write_json(destination / 'paths.json', dict(image_prefix=str(derived / 'images'),
        train_annotations=str(destination / 'annotations/instances_train.json'),
        validation_annotations=str(destination / 'annotations/instances_validation.json'),
        test_annotations=str(destination / 'annotations/instances_test.json')))
    report.update(restored_images=10, restored_train_annotations=restored_annotations['train'],
                  restored_test_annotations=restored_annotations['test'],
                  source_annotation_hashes={s: summary['splits'][s]['annotation_sha256'] for s in ('train', 'test')},
                  original_test_preserved=True, environment_variants='EXCLUDED',
                  derived_backup=str(backup))
    write_json(destination / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--derived', type=Path, required=True)
    parser.add_argument('--approvals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.derived, read(args.approvals), args.output)
