"""Package the reviewed 25-train/5-validation exclusions, bound to v10."""
import json
import zipfile
from collections import Counter
from pathlib import Path
from .apply_ruod_historical22 import read, digest
from .apply_ruod_consolidated import revise

def main():
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-v10-expanded-review'; inputs=repo/'outputs/ruod-v10-refresh-inputs'
    review=read(folder/'visual_review.json'); summary=read(folder/'review_summary.json')
    for name,sha in review['evidence_hashes'].items():
        if digest(folder/name)!=sha: raise ValueError('Review evidence changed')
    if digest(inputs/'reviewed_edges.json')!=review['prior_edges_sha256']:
        raise ValueError('Historical graph changed')
    if digest(folder/'round7.json')!=summary['terminal_scan_sha256'] or read(folder/'round7.json')['candidate_links']!=0:
        raise ValueError('Expansion boundary changed')
    members={k:{f:r[f] for f in ('origin','id','file_name','source_sha256')} for k,r in review['members'].items()}
    prior=read(inputs/'reviewed_edges.json')
    edges=[dict(e,review_status='visually_reviewed',provenance='historical_review') for e in prior
           if e['left'] in members and e['right'] in members]
    for e in review['decisions']+summary['explicit_cross_sheet_bridges']:
        if e['review_status']!='visually_reviewed' or e['verdict']!='same_scene_family_supported':
            raise ValueError('Unapproved decision')
        edges.append(e)
    for r in members.values():
        if digest(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])!=r['source_sha256']:
            raise ValueError('Original changed')
    spec=read(inputs/'spec.json')
    m=dict(base_split='v10-reviewed',output_split='v11-reviewed',preserve_validation=False,
        members=members,reviewed_edges=edges,review_sha256=digest(folder/'visual_review.json'),
        review_summary_sha256=digest(folder/'review_summary.json'),proposed_actions=summary['proposed_actions'],
        expected_images=dict(train=7515,validation=842,test=4200),
        **{k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')})
    data={s:read(repo/'outputs/ruod-v10-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    result,actions=revise(data,m)
    if Counter(a['from_split'] for a in actions)!=dict(train=25,validation=5) or any(a['to_split']!='exclude' for a in actions):
        raise ValueError('Unexpected actions')
    validation_names={a['file_name'] for a in m['proposed_actions'] if a['from_split']=='validation'}
    if validation_names!={'004707.jpg','004708.jpg','006314.jpg','006312.jpg','004433.jpg'}:
        raise ValueError('Unexpected validation exclusions')
    recovery={k:spec[k] for k in ('base_contracts','source_annotation_hashes','official_test_sha256')}; recovery['records']={}
    for origin in ('train','test'):
        ids={i['id'] for s,d in data.items() if ('test' if s=='test' else 'train')==origin for i in d['images']}
        rows=read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json'))
        recovery['records'][origin]={str(r['id']):{k:r[k] for k in ('file_name','file_sha256','stored_pixels_sha256')} for r in rows if r['id'] in ids}
        if len(recovery['records'][origin])!=len(ids): raise ValueError('Incomplete recovery evidence')
    payload={n:(repo/'tools'/n).read_bytes() for n in ('apply_ruod_consolidated.py','apply_ruod_historical22.py','restore_ruod_images.py','audit_ruod_dataset.py','apply_ruod_visual_families.py')}
    payload.update({'reviewed_manifest.json':json.dumps(m,indent=2).encode(),'recovery_manifest.json':json.dumps(recovery).encode(),
        'visual_review.json':json.dumps(dict(review,expanded_review_summary=summary),indent=2).encode()})
    proposal=repo/'outputs/ruod-v11-expanded30-proposal'; proposal.mkdir(exist_ok=True)
    (proposal/'reviewed_manifest.json').write_bytes(payload['reviewed_manifest.json'])
    archive=repo/'colab/ULGF_checkpoint02_expanded30.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in payload.items(): z.writestr(name,blob)
    cell=(repo/'colab/ULGF_checkpoint02_consolidated_cell.py').read_text(encoding='utf-8')
    replacements=[('v9','v11'),('v8','v10'),('V8','V10'),
        ('ruod-visual-families-v10-historical14','ruod-visual-families-v10-historical10'),
        ('ruod-visual-families-v11-consolidated','ruod-visual-families-v11-expanded30'),
        ('ULGF_checkpoint02_consolidated.zip','ULGF_checkpoint02_expanded30.zip'),
        ('d64619a7f7e45618ceb852cff66db3e97a8127ff70293988dfa754f2f9cb4431',digest(archive)),
        ('479 original hashes','115 original hashes'),('111 exclusions and nine moves','30 exclusions (25 train, five validation) and zero moves'),
        ("'train':7550,'validation':847","'train':7515,'validation':842"),
        ("summary['excluded_images']!=111","summary['excluded_images']!=30"),("summary['moved_images']!=9","summary['moved_images']!=0")]
    for a,b in replacements: cell=cell.replace(a,b)
    guard="""# Explicitly check both changed partitions, not only their total counts.
applied=json.loads((staging/'actions.json').read_text())
if sum(a['from_split']=='train' and a['to_split']=='exclude' for a in applied)!=25 or sum(a['from_split']=='validation' and a['to_split']=='exclude' for a in applied)!=5:
    raise RuntimeError('Unexpected partition-specific exclusions.')
# Staging location is temporary;"""
    cell=cell.replace('# Staging location is temporary;',guard)
    (repo/'colab/ULGF_checkpoint02_expanded30_cell.py').write_text(cell,encoding='utf-8')
    print('115 source hashes; 25 train + five validation exclusions; zero moves; expected 7515/842/4200.')
    print('ZIP SHA256:',digest(archive))

if __name__=='__main__': main()
