"""Build one v8-bound application/recovery package from completed visual review."""
import json
import zipfile
from pathlib import Path
from .apply_ruod_historical22 import read, digest
from .apply_ruod_consolidated import revise


def main():
    repo=Path(__file__).resolve().parents[1]
    review_root=repo/'outputs/ruod-v8-priority-review'
    inputs=repo/'outputs/ruod-v8-refresh-inputs'
    review=read(review_root/'visual_review.json')
    if digest(review_root/'inventory.json')!=review['inventory_sha256'] or digest(inputs/'reviewed_edges.json')!=review['prior_edges_sha256']:
        raise ValueError('Review evidence changed')
    edges=[]
    for e in read(inputs/'reviewed_edges.json'):
        edges.append(dict(e,review_status='visually_reviewed',provenance='previously_reviewed_family'))
    for e in review['decisions']+review['explicit_cross_sheet_bridges']:
        if e['verdict']!='same_scene_family_supported' or e['review_status']!='visually_reviewed':
            raise ValueError('Unapproved decision')
        edges.append(e)
    lookup={r['origin']+':'+str(r['id']):r for r in read(inputs/'fingerprints.json')}
    members={k:{f:lookup[k][f] for f in ('origin','id','file_name','source_sha256')}
             for k in sorted({e[s] for e in edges for s in ('left','right')})}
    for r in members.values():
        if digest(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])!=r['source_sha256']:
            raise ValueError('Reviewed original changed')
    spec=read(inputs/'spec.json')
    m=dict(base_split='v8-reviewed',output_split='v9-reviewed',members=members,reviewed_edges=edges,
        proposed_actions=review['proposed_actions'],expected_images=dict(train=7550,validation=847,test=4200),
        review_sha256=digest(review_root/'visual_review.json'),
        **{k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')})
    data={s:read(repo/'outputs/ruod-v8-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    result,actions=revise(data,m)
    assert sum(a['to_split']=='exclude' for a in actions)==111
    assert sum(a['to_split']!='exclude' for a in actions)==9
    recovery={k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')}
    recovery['records']={}
    for origin in ('train','test'):
        ids={i['id'] for s,d in data.items() if ('test' if s=='test' else 'train')==origin for i in d['images']}
        rows=read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json'))
        recovery['records'][origin]={str(r['id']):{k:r[k] for k in ('file_name','file_sha256','stored_pixels_sha256')} for r in rows if r['id'] in ids}
        assert len(recovery['records'][origin])==len(ids)
    payload={n:(repo/'tools'/n).read_bytes() for n in ('apply_ruod_consolidated.py','apply_ruod_historical22.py','restore_ruod_images.py','audit_ruod_dataset.py','apply_ruod_visual_families.py')}
    payload.update({'reviewed_manifest.json':json.dumps(m,indent=2).encode(),
        'recovery_manifest.json':json.dumps(recovery).encode(),
        'visual_review.json':(review_root/'visual_review.json').read_bytes()})
    out=repo/'outputs/ruod-v9-consolidated-proposal'
    out.mkdir(exist_ok=True)
    (out/'reviewed_manifest.json').write_bytes(payload['reviewed_manifest.json'])
    archive=repo/'colab/ULGF_checkpoint02_consolidated.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in payload.items(): z.writestr(name,blob)
    print('PACKAGE SHA256:',digest(archive))
    print('Evidence source hashes:',len(members),'Exclusions: 111; moves: 9; expected:',m['expected_images'])


if __name__=='__main__': main()
