"""Apply reviewed historical-family exclusions, never candidate-score decisions."""
import argparse
import json
import hashlib
import shutil
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def contract(data):
    payload = {k: sorted(data[k], key=lambda x: x['id'])
               for k in ('images', 'annotations', 'categories')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def revise(data, manifest, expected_exclusions=22):
    for split in ('train', 'validation', 'test'):
        if contract(data[split]) != manifest['base_contracts'][split]:
            raise ValueError('Base contract mismatch: ' + split)
    members = manifest['members']
    graph = {}
    for edge in manifest['reviewed_edges']:
        if edge['review_status'] != 'visually_reviewed':
            raise ValueError('Unreviewed edge')
        a, b = edge['left'], edge['right']
        if a not in members or b not in members:
            raise ValueError('Missing edge member')
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    excluded = manifest['proposed_exclusions']
    if len(excluded) != expected_exclusions or len(set(excluded)) != expected_exclusions:
        raise ValueError('Expected {} unique reviewed exclusions'.format(expected_exclusions))
    train_ids = {i['id'] for i in data['train']['images']}
    test_ids = {i['id'] for i in data['test']['images']}
    ids = set()
    for key in excluded:
        member = members[key]
        if member['origin'] != 'train' or member['id'] not in train_ids:
            raise ValueError('Exclusion not in current training split: ' + key)
        seen, pending = set(), [key]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(graph.get(node, set()) - seen)
        if not any(members[k]['origin'] == 'test' and members[k]['id'] in test_ids for k in seen):
            raise ValueError('No explicit reviewed path to official test: ' + key)
        ids.add(member['id'])
    if len(ids) != expected_exclusions:
        raise ValueError('Duplicate source identities')
    result = dict(data)
    result['train'] = dict(data['train'],
        images=[i for i in data['train']['images'] if i['id'] not in ids],
        annotations=[a for a in data['train']['annotations'] if a['image_id'] not in ids])
    if {a['category_id'] for a in result['train']['annotations']} != {c['id'] for c in result['train']['categories']}:
        raise ValueError('A training class was lost')
    return result


def apply(source, base, output, manifest_path, expected_exclusions=22):
    source, base, output = map(lambda p: Path(p).resolve(), (source, base, output))
    for protected in (source, base):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be separate from source/base')
    if output.exists():
        raise ValueError('Output exists; do not overwrite')
    m = read(manifest_path)
    data = {s: read(base/'annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    result = revise(data, m, expected_exclusions)
    if digest(base/'annotations/instances_test.json') != m['official_test_sha256']:
        raise ValueError('Official test bytes changed')
    for split, expected in m['source_annotation_hashes'].items():
        if digest(source/'RUOD_ANN'/('instances_'+split+'.json')) != expected:
            raise ValueError('Original annotations changed')
    for key, member in m['members'].items():
        if member['origin'] not in ('train','test'):
            raise ValueError('Invalid source origin')
        root = source/'RUOD_pic'/member['origin']
        if digest(checked_path(root, member['file_name'])) != member['source_sha256']:
            raise ValueError('Source image hash mismatch: '+key)
    prefix = Path(read(base/'paths.json')['image_prefix'])
    for dataset in result.values():
        for row in dataset['images']:
            if not checked_path(prefix, row['file_name']).is_file():
                raise FileNotFoundError('Derived images missing; recover images without repeating audit')
    # All input checks precede output creation; a failure after this point leaves
    # no PASS report, and subsequent execution refuses to overwrite it.
    (output/'annotations').mkdir(parents=True)
    (output/'annotations/instances_train.json').write_text(json.dumps(result['train'], indent=2), encoding='utf-8')
    for split in ('validation','test'):
        src = base/'annotations'/('instances_'+split+'.json')
        dst = output/'annotations'/src.name
        shutil.copy2(src, dst)
        if digest(src) != digest(dst):
            raise ValueError('Byte-preserving copy failed')
    paths = dict(image_prefix=str(prefix), **{s+'_annotations':str(output/'annotations'/('instances_'+s+'.json')) for s in result})
    (output/'paths.json').write_text(json.dumps(paths, indent=2))
    shutil.copy2(manifest_path, output/'reviewed_manifest.json')
    report = dict(status='REVIEWED_{}_EXCLUSIONS_PASS'.format(expected_exclusions), images={s:len(d['images']) for s,d in result.items()},
        excluded_images=expected_exclusions, moved_images=0, original_hashes_checked=len(m['members']),
        original_file_hashes='PASS', official_test_unchanged=True, validation_unchanged=True,
        official_test_sha256=digest(output/'annotations/instances_test.json'),
        manifest_sha256=digest(manifest_path), base_split_preserved=str(base),
        training_ready=False, training_started=False,
        limitation='Remaining candidates require review; no general near-duplicate clearance.')
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source','base','output','manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-exclusions',type=int,choices=(14,22),default=22)
    args = parser.parse_args()
    apply(args.source, args.base, args.output, args.manifest, args.expected_exclusions)
