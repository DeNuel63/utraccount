"""Apply only hash-bound, explicitly reviewed family decisions to a fresh split."""
import argparse
import json
import shutil
from pathlib import Path
try:
    from .apply_ruod_historical22 import read, digest, contract, checked_path
except ImportError:
    from apply_ruod_historical22 import read, digest, contract, checked_path

SPLITS = ('train', 'validation', 'test')


def revise(data, manifest):
    for s in SPLITS:
        if contract(data[s]) != manifest['base_contracts'][s]:
            raise ValueError('Base contract mismatch: ' + s)
    current = {}
    for s in SPLITS:
        for row in data[s]['images']:
            key = ('test' if s == 'test' else 'train') + ':' + str(row['id'])
            if key in current:
                raise ValueError('Duplicate source identity')
            current[key] = s
    graph = {}
    for key, member in manifest['members'].items():
        if member['origin'] not in ('train', 'test') or key != member['origin'] + ':' + str(member['id']):
            raise ValueError('Invalid member identity')
    for edge in manifest['reviewed_edges']:
        if edge['review_status'] != 'visually_reviewed':
            raise ValueError('Unreviewed edge')
        a, b = edge['left'], edge['right']
        if a not in manifest['members'] or b not in manifest['members']:
            raise ValueError('Missing evidence member')
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    actions, seen = [], set()
    for start in sorted(graph):
        if start in seen:
            continue
        group, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in group:
                continue
            group.add(node)
            pending.extend(graph[node] - group)
        seen.update(group)
        live = {k: current[k] for k in group if k in current}
        target = 'exclude' if 'test' in live.values() else ('validation' if 'validation' in live.values() else 'train')
        actions.extend(dict(member=k, from_split=s, to_split=target)
                       for k, s in sorted(live.items()) if s != 'test' and s != target)
    def signature(rows):
        return sorted((r['member'], r['from_split'], r['to_split']) for r in rows)
    if signature(actions) != signature(manifest['proposed_actions']):
        raise ValueError('Exact proposed actions differ from reviewed family closure')
    targets = {a['member']: a['to_split'] for a in actions}
    result = {'test': data['test']}
    for target in ('train', 'validation'):
        images, annotations = [], []
        for source in ('train', 'validation'):
            ids = {i['id'] for i in data[source]['images']
                   if targets.get('train:' + str(i['id']), source) == target}
            images.extend(i for i in data[source]['images'] if i['id'] in ids)
            annotations.extend(a for a in data[source]['annotations'] if a['image_id'] in ids)
        if len({a['id'] for a in annotations}) != len(annotations):
            raise ValueError('Annotation IDs collide')
        result[target] = dict(data[target], images=sorted(images, key=lambda i:i['id']),
                              annotations=sorted(annotations, key=lambda a:a['id']))
        if {a['category_id'] for a in annotations} != {c['id'] for c in data[target]['categories']}:
            raise ValueError('Class coverage lost')
    if {s: len(result[s]['images']) for s in SPLITS} != manifest['expected_images']:
        raise ValueError('Unexpected output counts')
    return result, actions


def apply(source, base, output, manifest_path):
    source, base, output = [Path(p).resolve() for p in (source, base, output)]
    for protected in (source, base):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Output must be separate')
    if output.exists():
        raise ValueError('Output exists; no overwrite')
    m = read(manifest_path)
    data = {s: read(base/'annotations'/('instances_'+s+'.json')) for s in SPLITS}
    result, actions = revise(data, m)
    preserve_validation = m.get('preserve_validation', False)
    if preserve_validation and result['validation'] != data['validation']:
        # Ordering is not a membership change; compare canonical contracts.
        if contract(result['validation']) != contract(data['validation']):
            raise ValueError('Validation must remain unchanged')
    if digest(base/'annotations/instances_test.json') != m['official_test_sha256']:
        raise ValueError('Official test bytes changed')
    for s, sha in m['source_annotation_hashes'].items():
        if digest(source/'RUOD_ANN'/('instances_'+s+'.json')) != sha:
            raise ValueError('Source annotation hash mismatch')
    for key, row in m['members'].items():
        if digest(checked_path(source/'RUOD_pic'/row['origin'], row['file_name'])) != row['source_sha256']:
            raise ValueError('Source image hash mismatch: '+key)
    prefix = Path(read(base/'paths.json')['image_prefix'])
    for d in result.values():
        for i in d['images']:
            if not checked_path(prefix, i['file_name']).is_file():
                raise FileNotFoundError('Missing derived image: '+i['file_name'])
    (output/'annotations').mkdir(parents=True)
    for s in ('train','validation'):
        target = output/'annotations'/('instances_'+s+'.json')
        if s == 'validation' and preserve_validation:
            original = base/'annotations/instances_validation.json'
            shutil.copy2(original, target)
            if digest(original) != digest(target):
                raise ValueError('Validation byte copy mismatch')
        else:
            target.write_text(json.dumps(result[s],indent=2),encoding='utf-8')
    shutil.copy2(base/'annotations/instances_test.json',output/'annotations/instances_test.json')
    if digest(output/'annotations/instances_test.json') != m['official_test_sha256']:
        raise ValueError('Test copy mismatch')
    paths = dict(image_prefix=str(prefix), **{s+'_annotations':str(output/'annotations'/('instances_'+s+'.json')) for s in SPLITS})
    (output/'paths.json').write_text(json.dumps(paths,indent=2))
    shutil.copy2(manifest_path,output/'reviewed_manifest.json')
    (output/'actions.json').write_text(json.dumps(actions,indent=2))
    report = dict(status='CONSOLIDATED_REVIEWED_APPLICATION_PASS',images=m['expected_images'],
        excluded_images=sum(a['to_split']=='exclude' for a in actions),
        moved_images=sum(a['to_split']!='exclude' for a in actions),original_hashes_checked=len(m['members']),
        original_file_hashes='PASS',official_test_unchanged=True,official_test_sha256=m['official_test_sha256'],
        manifest_sha256=digest(manifest_path),base_split_preserved=str(base),unreviewed_candidates_applied=0,
        training_ready=False,training_started=False,limitation='Remaining weaker candidates and exclusion coverage still require review.')
    if preserve_validation:
        report.update(validation_unchanged=True, validation_sha256=digest(output/'annotations/instances_validation.json'))
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('source','base','output','manifest'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    apply(a.source,a.base,a.output,a.manifest)
