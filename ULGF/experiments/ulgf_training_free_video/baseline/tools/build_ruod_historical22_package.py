"""Bind the completed local visual review to v6 and historical test paths."""
from pathlib import Path
import json
import zipfile
from .apply_ruod_visual_families import read, write, revise, contract, digest_file
from .build_ruod_visual_manifest import reconstruct_base
from .apply_ruod_historical22 import revise as revise22


def main():
    repo = Path(__file__).resolve().parents[1]
    folder = repo/'outputs/ruod-historical22-review-v6'
    review, inventory = read(folder/'visual_review.json'), read(folder/'inventory.json')
    if digest_file(folder/'inventory.json') != review['inventory_sha256']:
        raise ValueError('Reviewed inventory changed')
    folders = ['ruod-visual-family-proposal']+['ruod-visual-family-proposal-batch'+s for s in ('02','03','04','05')]
    prior = [read(repo/'outputs'/f/'reviewed_visual_families.json') for f in folders]
    data = reconstruct_base(repo.parent/'RUOD', repo/'outputs/ruod-local-audit')
    members, edges = {}, []
    for manifest in prior:
        data = revise(data, manifest)[0]
        for edge in manifest['reviewed_edges']:
            ends = []
            for side in ('left','right'):
                row = manifest['members'][edge[side]]
                key = row['origin']+':'+str(row['id'])
                members[key] = {k:row[k] for k in ('origin','id','file_name','source_sha256')}
                ends.append(key)
            edges.append(dict(left=ends[0], right=ends[1], review_status='visually_reviewed', provenance='prior_review'))
    decisions = {d['family_index']:d for d in review['decisions']}
    for group in inventory:
        decision = decisions[group['family_index']]
        if decision['verdict'] != 'same_scene_family_supported' or decision['reviewed_links'] != group['links']:
            raise ValueError('Review does not match inventory')
        for key, row in group['members'].items():
            members[key] = {k:row[k] for k in ('origin','id','file_name','source_sha256')}
        for edge in group['links']:
            ends = [edge[s]['origin']+':'+str(edge[s]['id']) for s in ('left','right')]
            edges.append(dict(left=ends[0],right=ends[1],review_status='visually_reviewed',provenance='historical22',finding=decision['finding']))
    # Include only connected historical components needed to establish the
    # reviewed exclusions' test ancestry, never unrelated candidate bridges.
    needed = set(review['proposed_exclusions'])
    while True:
        expanded = needed | {k for e in edges if {e['left'],e['right']} & needed for k in (e['left'],e['right'])}
        if expanded == needed:
            break
        needed = expanded
    manifest = dict(base_split='v6-reviewed', base_contracts={s:contract(d) for s,d in data.items()},
        source_annotation_hashes=prior[0]['source_annotation_hashes'],
        official_test_sha256='554b3a631fccf53f651aee8b8e0fcccf2c4a3763325d86ab9807449a7f4cff8f',
        members={k:members[k] for k in sorted(needed)},
        reviewed_edges=[e for e in edges if e['left'] in needed],
        proposed_exclusions=review['proposed_exclusions'],
        review_sha256=digest_file(folder/'visual_review.json'))
    result = revise22(data, manifest)
    assert {s:len(d['images']) for s,d in result.items()} == dict(train=7670,validation=852,test=4200)
    for member in manifest['members'].values():
        if digest_file(repo.parent/'RUOD/RUOD_pic'/member['origin']/member['file_name']) != member['source_sha256']:
            raise ValueError('Local source hash mismatch')
    output = repo/'outputs/ruod-historical22-proposal'
    write(output/'reviewed_manifest.json',manifest)
    recovery = {key:manifest[key] for key in ('base_contracts','source_annotation_hashes','official_test_sha256')}
    recovery['records'] = {}
    for origin in ('train','test'):
        ids = {i['id'] for split,d in data.items() if ('test' if split=='test' else 'train')==origin for i in d['images']}
        records = read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json'))
        recovery['records'][origin] = {str(r['id']):{k:r[k] for k in
            ('file_name','file_sha256','stored_pixels_sha256')} for r in records if r['id'] in ids}
        if len(recovery['records'][origin]) != len(ids):
            raise ValueError('Missing audited recovery records')
    with zipfile.ZipFile(repo/'colab/ULGF_checkpoint01.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(repo/'tools/apply_ruod_historical22.py','apply_ruod_historical22.py')
        archive.write(output/'reviewed_manifest.json','reviewed_manifest.json')
        for name in ('restore_ruod_images.py','audit_ruod_dataset.py','apply_ruod_visual_families.py'):
            archive.write(repo/'tools'/name,name)
        archive.writestr('recovery_manifest.json',json.dumps(recovery))
    print('Checkpoint 1 packaged; {} evidence hashes; expected 7670/852/4200. No Colab changes.'.format(len(manifest['members'])))


if __name__ == '__main__':
    main()
