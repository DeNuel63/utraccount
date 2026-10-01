"""Build and package explicit batch-04 decisions against verified v4 lineage."""
from collections import Counter
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, revise, digest_file
from .build_ruod_visual_manifest import reconstruct_base
from .build_ruod_batch02_manifest import assemble_manifest

def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo.parent / 'RUOD'
    prior = [repo / 'outputs' / name / 'reviewed_visual_families.json' for name in
             ('ruod-visual-family-proposal','ruod-visual-family-proposal-batch02',
              'ruod-visual-family-proposal-batch03')]
    hashes = read(prior[0])['source_annotation_hashes']
    for split, expected in hashes.items():
        if digest_file(source / 'RUOD_ANN' / ('instances_'+split+'.json')) != expected:
            raise ValueError('Original annotations changed')
    base = reconstruct_base(source, repo / 'outputs/ruod-local-audit')
    for path in prior:
        previous = read(path)
        if previous['source_annotation_hashes'] != hashes:
            raise ValueError('Source lineage differs')
        base = revise(base, previous)[0]
    if {s:len(d['images']) for s,d in base.items()} != dict(train=7751,validation=857,test=4200):
        raise ValueError('V4 counts mismatch')
    folder = repo / 'outputs/ruod-near-review-batch04-v4'
    review = read(folder / 'visual_review.json')
    manifest = assemble_manifest(base,read(folder/'batch_candidates.json'),review,
        digest_file(folder/'batch_candidates.json'),hashes,batch_id='04',base_split='v4-reviewed',
        allow_training_families=True)
    manifest['prior_review_manifest_hashes'] = [digest_file(p) for p in prior]
    for member in manifest['members'].values():
        if digest_file(source/'RUOD_pic'/member['origin']/member['file_name']) != member['source_sha256']:
            raise ValueError('Reviewed source hash mismatch')
    result, actions, summary = revise(base, manifest)
    if (len(manifest['members']),len(manifest['reviewed_edges']),len(manifest['families'])) != (52,45,12):
        raise ValueError('Unexpected batch scope')
    if Counter((a['from_split'],a['to_split']) for a in actions) != {('train','exclude'):19,('validation','exclude'):2,('train','validation'):13}:
        raise ValueError('Unexpected actions')
    if {(a['member'],a['to_split']) for a in actions} != {(a['member'],a['to_split']) for a in review['proposed_actions']}:
        raise ValueError('Actions differ from visual decisions')
    if summary['images'] != dict(train=7719,validation=868,test=4200) or result['test'] != base['test']:
        raise ValueError('Output counts/test mismatch')
    output = repo / 'outputs/ruod-visual-family-proposal-batch04'
    write(output/'reviewed_visual_families.json',manifest)
    write(output/'proposed_actions.json',actions)
    write(output/'proposal_summary.json',summary)
    with zipfile.ZipFile(repo/'colab/RUOD_visual_families_batch04.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(repo/'tools/apply_ruod_visual_families.py','apply_ruod_visual_families.py')
        archive.write(output/'reviewed_visual_families.json','reviewed_visual_families.json')
    print(summary)

if __name__ == '__main__':
    main()
