"""Package the v7-bound reviewed application phase of checkpoint 2."""
from pathlib import Path
import zipfile
import json
from .apply_ruod_visual_families import read, write, digest_file, contract
from .apply_ruod_historical22 import revise

def main():
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-historical14-review-v7'
    review=read(folder/'visual_review.json')
    if digest_file(folder/'inventory.json')!=review['inventory_sha256']:
        raise ValueError('Review inventory changed')
    inputs=repo/'outputs/ruod-v7-refresh-inputs'
    if digest_file(inputs/'reviewed_edges.json')!=review['prior_reviewed_edges_sha256']:
        raise ValueError('Historical evidence changed')
    prior={frozenset((e['left'],e['right'])) for e in read(inputs/'reviewed_edges.json')}
    edges=[]
    for d in review['decisions']:
        if d['review_status']!='visually_reviewed' or d['verdict']!='same_scene_family_supported':
            raise ValueError('Unapproved decision')
        edges.append(dict(left=d['left'],right=d['right'],review_status='visually_reviewed',finding=d['finding']))
        path=d['anchor_to_test_reviewed_path']
        if path[0]!=d['right'] or not path[-1].startswith('test:'):
            raise ValueError('Invalid ancestry')
        for a,b in zip(path,path[1:]):
            if frozenset((a,b)) not in prior:
                raise ValueError('Unreviewed ancestry edge')
            edges.append(dict(left=a,right=b,review_status='visually_reviewed',provenance='prior_review'))
    lookup={r['origin']+':'+str(r['id']):r for r in read(inputs/'fingerprints.json')}
    needed={e[s] for e in edges for s in ('left','right')}
    members={k:{f:lookup[k][f] for f in ('origin','id','file_name','source_sha256')} for k in sorted(needed)}
    for r in members.values():
        if digest_file(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])!=r['source_sha256']:
            raise ValueError('Source image changed')
    spec=read(inputs/'spec.json')
    m=dict(base_split='v7-reviewed',members=members,reviewed_edges=edges,
        proposed_exclusions=review['proposed_exclusions'],review_sha256=digest_file(folder/'visual_review.json'),
        **{k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')})
    data={s:read(repo/'outputs/ruod-v7-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    result=revise(data,m,14)
    if {s:len(d['images']) for s,d in result.items()}!=dict(train=7656,validation=852,test=4200):
        raise ValueError('Unexpected output scope')
    output=repo/'outputs/ruod-historical14-proposal'
    write(output/'reviewed_manifest.json',m)
    recovery={k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')}
    recovery['records']={}
    for origin in ('train','test'):
        ids={i['id'] for s,d in data.items() if ('test' if s=='test' else 'train')==origin for i in d['images']}
        rows=read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json'))
        recovery['records'][origin]={str(r['id']):{k:r[k] for k in ('file_name','file_sha256','stored_pixels_sha256')} for r in rows if r['id'] in ids}
        if len(recovery['records'][origin])!=len(ids):
            raise ValueError('Incomplete recovery evidence')
    with zipfile.ZipFile(repo/'colab/ULGF_checkpoint02_apply.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('apply_ruod_historical22.py','restore_ruod_images.py','audit_ruod_dataset.py','apply_ruod_visual_families.py'):
            z.write(repo/'tools'/name,name)
        z.write(output/'reviewed_manifest.json','reviewed_manifest.json')
        z.writestr('recovery_manifest.json',json.dumps(recovery))
    print('14 exclusions; {} evidence hashes; proposed 7656/852/4200; no application.'.format(len(members)))

if __name__=='__main__':
    main()
