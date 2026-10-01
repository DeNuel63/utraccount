"""Build v9->v10 package and matching recovery-safe single Colab cell."""
import json
import zipfile
from pathlib import Path
from .apply_ruod_historical22 import read, digest
from .apply_ruod_consolidated import revise

def main():
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-v9-historical-review'; inputs=repo/'outputs/ruod-v9-refresh-inputs'
    review=read(folder/'visual_review.json'); inventory=read(folder/'inventory.json')
    if digest(folder/'inventory.json')!=review['inventory_sha256'] or digest(inputs/'reviewed_edges.json')!=review['prior_edges_sha256']:
        raise ValueError('Review evidence changed')
    prior={frozenset((e['left'],e['right'])) for e in read(inputs/'reviewed_edges.json')}
    edges=[]
    for d in review['decisions']:
        if d['review_status']!='visually_reviewed' or d['verdict']!='same_scene_family_supported': raise ValueError('Unreviewed decision')
        edges.append(d)
        path=d['anchor_to_test_reviewed_path']
        if path[0]!=d['right'] or not path[-1].startswith('test:'): raise ValueError('Invalid ancestry')
        for a,b in zip(path,path[1:]):
            if frozenset((a,b)) not in prior: raise ValueError('Unreviewed ancestry')
            edges.append(dict(left=a,right=b,review_status='visually_reviewed',provenance='historical_review'))
    for d in review['explicit_cross_sheet_bridges']:
        if d['review_status']!='visually_reviewed' or d['verdict']!='same_scene_family_supported': raise ValueError('Unreviewed bridge')
        edges.append(d)
    members={k:{f:r[f] for f in ('origin','id','file_name','source_sha256')} for k,r in inventory['members'].items()}
    for r in members.values():
        if digest(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])!=r['source_sha256']: raise ValueError('Source changed')
    spec=read(inputs/'spec.json')
    m=dict(base_split='v9-reviewed',output_split='v10-reviewed',preserve_validation=True,
        members=members,reviewed_edges=edges,review_sha256=digest(folder/'visual_review.json'),
        proposed_actions=[dict(member=k,from_split='train',to_split='exclude') for k in review['proposed_exclusions']],
        expected_images=dict(train=7540,validation=847,test=4200),
        **{k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')})
    data={s:read(repo/'outputs/ruod-v9-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    result,actions=revise(data,m)
    assert len(actions)==10 and all(a['from_split']=='train' and a['to_split']=='exclude' for a in actions)
    recovery={k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')}; recovery['records']={}
    for origin in ('train','test'):
        ids={i['id'] for s,d in data.items() if ('test' if s=='test' else 'train')==origin for i in d['images']}
        rows=read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json'))
        recovery['records'][origin]={str(r['id']):{k:r[k] for k in ('file_name','file_sha256','stored_pixels_sha256')} for r in rows if r['id'] in ids}
        assert len(recovery['records'][origin])==len(ids)
    payload={n:(repo/'tools'/n).read_bytes() for n in ('apply_ruod_consolidated.py','apply_ruod_historical22.py','restore_ruod_images.py','audit_ruod_dataset.py','apply_ruod_visual_families.py')}
    payload.update({'reviewed_manifest.json':json.dumps(m,indent=2).encode(),'recovery_manifest.json':json.dumps(recovery).encode(),
                    'visual_review.json':(folder/'visual_review.json').read_bytes()})
    proposal=repo/'outputs/ruod-v10-historical10-proposal'; proposal.mkdir(exist_ok=True)
    (proposal/'reviewed_manifest.json').write_bytes(payload['reviewed_manifest.json'])
    archive=repo/'colab/ULGF_checkpoint02_historical10.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in payload.items(): z.writestr(name,blob)
    # Generate the new cell from the tested recovery-safe template, preserving
    # the old deliverable and binding this cell to this exact new ZIP.
    cell=(repo/'colab/ULGF_checkpoint02_consolidated_cell.py').read_text(encoding='utf-8')
    replacements=[('v9','v10'),('v8','v9'),('V8','V9'),
        ('ruod-visual-families-v9-historical14','ruod-visual-families-v9-consolidated'),
        ('ruod-visual-families-v10-consolidated','ruod-visual-families-v10-historical10'),
        ('ULGF_checkpoint02_consolidated.zip','ULGF_checkpoint02_historical10.zip'),
        ('d64619a7f7e45618ceb852cff66db3e97a8127ff70293988dfa754f2f9cb4431',digest(archive)),
        ('479 original hashes','30 original hashes'),('111 exclusions and nine moves','ten exclusions and zero moves'),
        ("'train':7550","'train':7540"),("summary['excluded_images']!=111","summary['excluded_images']!=10"),
        ("summary['moved_images']!=9","summary['moved_images']!=0"),
        ('Validation membership intentionally updated.','Validation also byte-for-byte unchanged.')]
    for a,b in replacements: cell=cell.replace(a,b)
    cell=cell.replace("# Staging location is temporary;", "if not summary.get('validation_unchanged'):\n    raise RuntimeError('Validation preservation not confirmed.')\n# Staging location is temporary;")
    (repo/'colab/ULGF_checkpoint02_historical10_cell.py').write_text(cell,encoding='utf-8')
    print('30 source hashes; ten exclusions; zero moves; expected 7540/847/4200. ZIP SHA:',digest(archive))

if __name__=='__main__': main()
