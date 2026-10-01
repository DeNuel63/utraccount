"""Apply only reviewed, hash-bound visual-family edges to a NEW split version."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def contract(data):
    payload = {k: sorted(data[k], key=lambda r: r['id']) for k in ('images', 'annotations', 'categories')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def families_from_edges(edges):
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    for edge in edges:
        if edge.get('review_status') != 'visually_reviewed':
            raise ValueError('An unreviewed edge was supplied')
        parent[find(edge['left'])] = find(edge['right'])
    groups = {}
    for key in sorted(parent):
        groups.setdefault(find(key), []).append(key)
    return sorted(groups.values(), key=lambda group: tuple(group))


def revise(datasets, manifest):
    for split in ('train', 'validation', 'test'):
        if contract(datasets[split]) != manifest['base_contracts'][split]:
            raise ValueError('Base split differs from reviewed split: ' + split)
    groups = families_from_edges(manifest['reviewed_edges'])
    if groups != [family['members'] for family in manifest['families']]:
        raise ValueError('Family manifest differs from reviewed-edge closure')
    nodes = {split+':'+str(i['id']): (split, i) for split, data in datasets.items() for i in data['images']}
    if set(manifest['members']) != {key for group in groups for key in group}:
        raise ValueError('Unexpected reviewed members')
    assignments = {key: value[0] for key, value in nodes.items()}
    actions = []
    for index, group in enumerate(groups, 1):
        if not set(group) <= set(nodes):
            raise ValueError('Reviewed member missing from base split')
        splits = {nodes[key][0] for key in group}
        target = 'exclude' if 'test' in splits else ('validation' if 'validation' in splits else 'train')
        for key in group:
            old = nodes[key][0]
            new = 'test' if old == 'test' else target
            assignments[key] = new
            if old != new:
                actions.append(dict(member=key, image_id=nodes[key][1]['id'], from_split=old,
                                    to_split=new, family_id='VF{:03d}'.format(index),
                                    reason='reviewed_family_touches_test' if target=='exclude'
                                    else 'keep_reviewed_family_together'))
    # IDs in train and validation share the original training namespace.
    combined_images = datasets['train']['images'] + datasets['validation']['images']
    combined_anns = datasets['train']['annotations'] + datasets['validation']['annotations']
    if len({i['id'] for i in combined_images}) != len(combined_images):
        raise ValueError('Train/validation IDs overlap')
    result = {'test': datasets['test']}
    for split in ('train', 'validation'):
        ids = {nodes[key][1]['id'] for key, target in assignments.items()
               if target==split and nodes[key][0] != 'test'}
        result[split] = dict(datasets[split],
                            images=sorted([i for i in combined_images if i['id'] in ids], key=lambda i:i['id']),
                            annotations=sorted([a for a in combined_anns if a['image_id'] in ids], key=lambda a:a['id']))
    for group in groups:
        remaining = {assignments[key] for key in group} - {'exclude'}
        if len(remaining) > 1:
            raise ValueError('Reviewed family crosses output splits')
    counts = {}
    for split, data in result.items():
        count = Counter(a['category_id'] for a in data['annotations'])
        if set(count) != {c['id'] for c in data['categories']}:
            raise ValueError('Missing class in ' + split)
        counts[split] = {c['name']: count[c['id']] for c in data['categories']}
    report = dict(images={s:len(d['images']) for s,d in result.items()},
                  annotations_per_class=counts, reviewed_edges=len(manifest['reviewed_edges']),
                  reviewed_families=len(groups), reviewed_members=len(manifest['members']),
                  excluded_images=sum(a['to_split']=='exclude' for a in actions),
                  moved_images=sum(a['to_split']!='exclude' for a in actions),
                  reviewed_family_cross_split_check='PASS', official_test_unchanged=True,
                  unreviewed_candidates_applied=0, training_ready=False,
                  limitation='Only explicitly reviewed edges and their transitive closure are resolved; remaining candidates require review.')
    return result, actions, report


def apply(source, base, output, manifest_path):
    source, base, output = source.resolve(), base.resolve(), output.resolve()
    if output == base or base in output.parents or output in base.parents:
        raise ValueError('Use a separate new split directory')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from original data')
    if output.exists():
        raise ValueError('Output already exists; preserve it and use a fresh destination')
    manifest = read(manifest_path)
    datasets = {s:read(base / 'annotations' / ('instances_'+s+'.json')) for s in ('train','validation','test')}
    # No output files are written until ALL manifest and original-hash checks pass.
    result, actions, report = revise(datasets, manifest)
    for split, expected in manifest['source_annotation_hashes'].items():
        if digest_file(source / 'RUOD_ANN' / ('instances_'+split+'.json')) != expected:
            raise ValueError('Original annotations changed: '+split)
    for key, member in manifest['members'].items():
        root = (source / 'RUOD_pic' / member['origin']).resolve()
        path = (root / member['file_name']).resolve()
        path.relative_to(root)
        if digest_file(path) != member['source_sha256']:
            raise ValueError('Reviewed source hash mismatch: '+key)
    image_prefix = Path(read(base / 'paths.json')['image_prefix'])
    for data in result.values():
        for item in data['images']:
            if not (image_prefix / item['file_name']).is_file():
                raise FileNotFoundError('Missing derived image: '+item['file_name'])
    for split in ('train','validation'):
        write(output / 'annotations' / ('instances_'+split+'.json'), result[split])
    # Preserve the previous official test manifest byte for byte, not just semantically.
    test_source = base / 'annotations/instances_test.json'
    test_target = output / 'annotations/instances_test.json'
    shutil.copy2(test_source, test_target)
    if digest_file(test_source) != digest_file(test_target):
        raise ValueError('Official test copy verification failed')
    write(output / 'actions.json', actions)
    write(output / 'reviewed_visual_families.json', manifest)
    write(output / 'paths.json', dict(image_prefix=str(image_prefix),
        **{s+'_annotations':str(output / 'annotations' / ('instances_'+s+'.json'))
           for s in ('train','validation','test')}))
    report.update(original_file_hashes='PASS', original_hashes_checked=len(manifest['members']),
                  base_split_preserved=str(base), manifest_sha256=digest_file(manifest_path),
                  official_test_sha256=digest_file(test_target))
    write(output / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    apply(args.source, args.base, args.output, args.manifest)
