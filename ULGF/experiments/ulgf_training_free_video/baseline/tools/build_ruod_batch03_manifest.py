"""Build only the recorded batch-03 decisions against the verified v3 lineage."""
import argparse
from pathlib import Path
from .apply_ruod_visual_families import read, write, revise, digest_file
from .build_ruod_visual_manifest import reconstruct_base
from .build_ruod_batch02_manifest import assemble_manifest


def build(source, audit, first, second, batch, output):
    first_manifest,second_manifest=read(first),read(second)
    annotation_hashes=second_manifest['source_annotation_hashes']
    if annotation_hashes!=first_manifest['source_annotation_hashes']:
        raise ValueError('Source annotation lineage differs')
    for split,expected in annotation_hashes.items():
        if digest_file(source/'RUOD_ANN'/('instances_'+split+'.json'))!=expected:
            raise ValueError('Source annotations changed')
    v1=reconstruct_base(source,audit)
    v2=revise(v1,first_manifest)[0]
    v3=revise(v2,second_manifest)[0]
    candidates=batch/'batch_candidates.json'
    manifest=assemble_manifest(v3,read(candidates),read(batch/'visual_review.json'),
        digest_file(candidates),annotation_hashes,batch_id='03',base_split='v3-reviewed')
    manifest['prior_review_manifest_hashes']=[digest_file(first),digest_file(second)]
    for member in manifest['members'].values():
        if digest_file(source/'RUOD_pic'/member['origin']/member['file_name'])!=member['source_sha256']:
            raise ValueError('Reviewed original changed: '+member['file_name'])
    result,actions,report=revise(v3,manifest)
    if (len(manifest['members']),len(manifest['reviewed_edges']),len(manifest['families']))!=(39,29,12):
        raise ValueError('Unexpected batch-03 scope')
    if len(actions)!=20 or any(a['from_split']!='train' or a['to_split']!='exclude' for a in actions):
        raise ValueError('Unexpected batch-03 action')
    if result['validation']!=v3['validation'] or result['test']!=v3['test']:
        raise ValueError('Validation or test changed')
    write(output/'reviewed_visual_families.json',manifest)
    write(output/'proposed_actions.json',actions)
    write(output/'proposal_summary.json',report)
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','first','second','batch','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.audit,args.first,args.second,args.batch,args.output)
