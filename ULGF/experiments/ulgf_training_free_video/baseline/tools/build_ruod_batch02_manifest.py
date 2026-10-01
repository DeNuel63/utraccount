"""Build batch-02 proposal against v2, using the recorded visual decisions only."""
import argparse
from pathlib import Path
try:
    from .apply_ruod_visual_families import read, write, contract, families_from_edges, revise, digest_file
    from .build_ruod_visual_manifest import reconstruct_base
except ImportError:
    from apply_ruod_visual_families import read, write, contract, families_from_edges, revise, digest_file
    from build_ruod_visual_manifest import reconstruct_base


def assemble_manifest(base, candidates, review, candidate_digest, annotation_hashes,
                      batch_id='02', base_split='v2-reviewed', allow_training_families=False):
    if review['batch_candidates_sha256'] != candidate_digest:
        raise ValueError('Candidate inventory changed after visual review')
    if review['review_status'] != 'BATCH_VISUAL_REVIEW_COMPLETE':
        raise ValueError('Visual review is incomplete')
    approved = {r['candidate_family']:r for r in review['reviews']}
    groups = {g['candidate_family']:g for g in candidates}
    if not set(approved) <= set(groups):
        raise ValueError('Reviewed group is absent')
    edges, members = [], {}
    for name, decision in approved.items():
        if decision['verdict'] not in ('same_scene_family_supported','near_identical_photo_supported'):
            raise ValueError('Unapproved visual finding')
        group = groups[name]
        shown = group['shown_members']
        if decision['members_reviewed'] != len(shown) or decision['links_reviewed'] != len(group['shown_edges']):
            raise ValueError('Review scope does not match displayed evidence')
        mapping = {}
        for original_key, row in shown.items():
            # Some v1 train members were moved into validation in v2.
            current_key = row['split']+':'+str(row['id'])
            mapping[original_key] = current_key
            member = {k:row[k] for k in ('split','origin','id','file_name','source_sha256')}
            if current_key in members and members[current_key] != member:
                raise ValueError('Conflicting member provenance')
            members[current_key] = member
        for edge in group['shown_edges']:
            edges.append(dict(left=mapping[edge['left']],right=mapping[edge['right']],
                              review_status='visually_reviewed',candidate_family=name,
                              finding=decision['finding']))
    families = [dict(family_id='VF{:03d}'.format(i),members=group,action='exclude_non_test_members')
                for i,group in enumerate(families_from_edges(edges),1)]
    if not allow_training_families and any(not any(members[key]['split']=='test' for key in family['members']) for family in families):
        raise ValueError('Unexpected non-test family in this reviewed batch')
    if allow_training_families:
        for family in families:
            if not any(members[key]['split']=='test' for key in family['members']):
                family['action']='keep_in_single_partition'
    return dict(version=2,batch=batch_id,review_date=review['review_date'],
                base_split=base_split,scope='Only the {} explicit batch-{} visually reviewed links'.format(len(edges),batch_id),
                base_contracts={s:contract(d) for s,d in base.items()},
                source_annotation_hashes=annotation_hashes,
                reviewed_candidate_inventory_sha256=candidate_digest,
                reviewed_edges=edges,members=members,families=families,
                policy='Preserve official test; exclude non-test members of reviewed test-linked families; keep training-only families in one partition (validation if already represented there).')


def build(source, audit, previous, batch, output):
    previous_manifest=read(previous)
    for split,expected in previous_manifest['source_annotation_hashes'].items():
        if digest_file(source/'RUOD_ANN'/('instances_'+split+'.json'))!=expected:
            raise ValueError('Original annotations changed')
    v1=reconstruct_base(source,audit)
    v2=revise(v1,previous_manifest)[0]
    candidates_path=batch/'batch_candidates.json'
    manifest=assemble_manifest(v2,read(candidates_path),read(batch/'visual_review.json'),
                               digest_file(candidates_path),previous_manifest['source_annotation_hashes'])
    for member in manifest['members'].values():
        if digest_file(source/'RUOD_pic'/member['origin']/member['file_name'])!=member['source_sha256']:
            raise ValueError('Reviewed source changed locally')
    result,actions,report=revise(v2,manifest)
    if report['excluded_images']!=28 or report['moved_images']!=0:
        raise ValueError('Unexpected batch-02 impact')
    write(output/'reviewed_visual_families.json',manifest)
    write(output/'proposed_actions.json',actions)
    write(output/'proposal_summary.json',report)
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','previous','batch','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.audit,args.previous,args.batch,args.output)
