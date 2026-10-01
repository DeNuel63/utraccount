"""Build batch-05 including reviewed bridge to the prior validation family."""
from collections import Counter
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, revise, digest_file
from .build_ruod_visual_manifest import reconstruct_base
from .build_ruod_batch02_manifest import assemble_manifest

def validate_proposal(base, manifest, review):
    result, actions, summary = revise(base, manifest)
    if (len(manifest['members']),len(manifest['reviewed_edges']),len(manifest['families'])) != (63,66,12):
        raise ValueError('Unexpected batch-05 scope')
    if Counter((a['from_split'],a['to_split']) for a in actions) != {('train','exclude'):27,('validation','exclude'):16}:
        raise ValueError('Unexpected batch-05 actions')
    if {(a['member'],a['to_split']) for a in actions} != {(a['member'],a['to_split']) for a in review['proposed_actions']}:
        raise ValueError('Actions differ from visual review')
    if summary['images'] != dict(train=7692,validation=852,test=4200) or result['test'] != base['test']:
        raise ValueError('Output counts/test mismatch')
    bridge={'test:2263','validation:7182'}
    if not any({e['left'],e['right']}==bridge for e in manifest['reviewed_edges']):
        raise ValueError('Explicit family bridge missing')
    family=next(f for f in manifest['families'] if bridge <= set(f['members']))
    if len(family['members']) != 23:
        raise ValueError('Prior family closure missing')
    return actions,summary

def main():
    repo=Path(__file__).resolve().parents[1]
    source=repo.parent/'RUOD'
    folders=['ruod-visual-family-proposal']+['ruod-visual-family-proposal-batch'+s for s in ('02','03','04')]
    prior=[repo/'outputs'/f/'reviewed_visual_families.json' for f in folders]
    hashes=read(prior[0])['source_annotation_hashes']
    for split,expected in hashes.items():
        if digest_file(source/'RUOD_ANN'/('instances_'+split+'.json')) != expected:
            raise ValueError('Original annotations changed')
    base=reconstruct_base(source,repo/'outputs/ruod-local-audit')
    for path in prior:
        previous=read(path)
        if previous['source_annotation_hashes'] != hashes:
            raise ValueError('Source lineage differs')
        base=revise(base,previous)[0]
    if {s:len(d['images']) for s,d in base.items()} != dict(train=7719,validation=868,test=4200):
        raise ValueError('V5 counts mismatch')
    folder=repo/'outputs/ruod-near-review-batch05-v5'
    review=read(folder/'visual_review.json')
    candidates=read(folder/'batch_candidates.json')
    manifest=assemble_manifest(base,candidates,review,digest_file(folder/'batch_candidates.json'),
                               hashes,batch_id='05',base_split='v5-reviewed')
    manifest['prior_review_manifest_hashes']=[digest_file(p) for p in prior]
    manifest['explicit_family_connection']=dict(left='test:2263',right='validation:7182',
        prior_family='B04F12',current_family='B05F06',prior_members_now_in_validation=14,
        reason='Reviewed matching turtle/reef scene; framing differs. Prior validation family now links to official test.')
    # Preserve link-level provenance for the inherited edges and explicit visual bridge.
    for edge in manifest['reviewed_edges']:
        group=next(g for g in candidates if g['candidate_family']==edge['candidate_family'])
        for original in group['shown_edges']:
            endpoints=[]
            for side in ('left','right'):
                member=group['shown_members'][original[side]]
                endpoints.append(member['split']+':'+str(member['id']))
            if endpoints==[edge['left'],edge['right']] and 'provenance' in original:
                edge['provenance']=original['provenance']
    for member in manifest['members'].values():
        if digest_file(source/'RUOD_pic'/member['origin']/member['file_name']) != member['source_sha256']:
            raise ValueError('Reviewed source hash mismatch')
    actions,summary=validate_proposal(base,manifest,review)
    output=repo/'outputs/ruod-visual-family-proposal-batch05'
    write(output/'reviewed_visual_families.json',manifest)
    write(output/'proposed_actions.json',actions)
    write(output/'proposal_summary.json',summary)
    with zipfile.ZipFile(repo/'colab/RUOD_visual_families_batch05.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(repo/'tools/apply_ruod_visual_families.py','apply_ruod_visual_families.py')
        archive.write(output/'reviewed_visual_families.json','reviewed_visual_families.json')
    print(summary)

if __name__=='__main__':
    main()
