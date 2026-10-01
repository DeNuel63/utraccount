"""Build a proposal ONLY from recorded visual reviews, never threshold candidates."""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
try:
    from .apply_ruod_visual_families import read, write, contract, families_from_edges, revise
    from .prepare_ruod_split import partition
    from .audit_ruod_dataset import export_filename
except ImportError:
    from apply_ruod_visual_families import read, write, contract, families_from_edges, revise
    from prepare_ruod_split import partition
    from audit_ruod_dataset import export_filename


def reconstruct_export_image(item, record, path):
    # Early audit versions did not record format. Inspect headers, not full pixels.
    with Image.open(path) as image:
        if list(image.size) != record['stored_size'] or image.mode != record['source_mode']:
            raise ValueError('Source header changed: '+str(path))
        name = export_filename(item['id'], image.mode, image.format)
    return dict(item, file_name=name, width=record['stored_size'][0], height=record['stored_size'][1])


def reconstruct_base(source, audit):
    """Reproduce v1 with the same JPEG/PNG naming policy as the real exporter."""
    datasets, records = {}, {}
    for split in ('train','test'):
        data = read(source / 'RUOD_ANN' / ('instances_'+split+'.json'))
        records[split] = read(audit / 'audit' / (split+'.json'))
        mapping = {r['id']:r for r in records[split]}
        with ThreadPoolExecutor(max_workers=4) as executor:
            data['images'] = list(executor.map(
                lambda i: reconstruct_export_image(i, mapping[i['id']], source/'RUOD_pic'/split/i['file_name']),
                data['images']))
        datasets[split] = data
    train, val, test = partition(datasets['train'], datasets['test'], records)[:3]
    return dict(train=train, validation=val, test=test)


def build(source, audit, screen, output):
    reviews = read(screen / 'visual_review.json')['reviews']
    fingerprints = read(screen / 'fingerprints.json')
    lookup = {r['split']+':'+str(r['id']):r for r in fingerprints}
    edges, members = [], {}
    for item in reviews:
        a,b = item['boundary'].split('__')
        left,right = a+':'+str(item['left_id']), b+':'+str(item['right_id'])
        edges.append(dict(left=left,right=right,review_status='visually_reviewed', finding=item['finding']))
        for key in (left,right):
            row = lookup[key]
            members[key] = {k:row[k] for k in ('split','origin','id','file_name','source_sha256')}
    families = []
    for index, group in enumerate(families_from_edges(edges), 1):
        splits = {members[key]['split'] for key in group}
        action = 'exclude_non_test_members' if 'test' in splits else 'keep_together_in_validation'
        families.append(dict(family_id='VF{:03d}'.format(index),members=group,action=action))
    base = reconstruct_base(source, audit)
    summary = read(audit / 'summary.json')
    manifest = dict(version=1, review_date='2026-09-23',
                    scope='24 explicit visually reviewed links only; no automatic candidate expansion',
                    source_annotation_hashes={s:summary['splits'][s]['annotation_sha256'] for s in ('train','test')},
                    base_contracts={s:contract(d) for s,d in base.items()}, members=members,
                    reviewed_edges=edges, families=families,
                    policy='Preserve official test; exclude non-test members of test-linked families; '
                           'keep remaining mixed train/validation families wholly in validation.')
    result, actions, report = revise(base, manifest)
    write(output / 'reviewed_visual_families.json',manifest)
    write(output / 'proposed_actions.json',actions)
    write(output / 'proposal_summary.json',report)
    print(report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','screen','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.audit,args.screen,args.output)
