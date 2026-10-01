"""Reconstruct verified v8 and package/read-only refresh with complete history."""
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, contract, digest_file
from .apply_ruod_historical22 import revise
from .refresh_ruod_v7 import refresh

def main():
    repo=Path(__file__).resolve().parents[1]
    old=repo/'outputs/ruod-v7-refresh-inputs'
    base=repo/'outputs/ruod-v8-refresh-base'
    bundle=repo/'outputs/ruod-v8-refresh-inputs'
    manifest_path=repo/'outputs/ruod-historical14-proposal/reviewed_manifest.json'
    if digest_file(manifest_path)!='ffda2699c961f397c44c2b780b044fe5e4e15def9fe58b36ee5f47676ccd36ab':
        raise ValueError('Manifest does not match successful Colab application')
    manifest=read(manifest_path)
    data={s:read(repo/'outputs/ruod-v7-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    data=revise(data,manifest,14)
    if {s:len(d['images']) for s,d in data.items()}!=dict(train=7656,validation=852,test=4200):
        raise ValueError('Wrong v8 counts')
    for s,d in data.items():
        write(base/'annotations'/('instances_'+s+'.json'),d)
    for name in ('fingerprints.json','exact_exclusions.json'):
        write(bundle/name,read(old/name))
    edges=read(old/'reviewed_edges.json')+manifest['reviewed_edges']
    edges=[dict(left=a,right=b) for a,b in sorted({tuple(sorted((e['left'],e['right']))) for e in edges})]
    write(bundle/'reviewed_edges.json',edges)
    spec=dict(read(old/'spec.json'),split='v8-reviewed',base_contracts={s:contract(d) for s,d in data.items()},
        applied_manifest_sha256=digest_file(manifest_path),
        payload_hashes={n:digest_file(bundle/n) for n in ('fingerprints.json','exact_exclusions.json','reviewed_edges.json')})
    write(bundle/'spec.json',spec)
    local=repo/'outputs/ruod-v8-local-refresh-inputs'
    for name in spec['payload_hashes']:
        write(local/name,read(bundle/name))
    write(local/'spec.json',dict(spec,official_test_sha256=digest_file(base/'annotations/instances_test.json')))
    refresh(base,local,repo.parent/'RUOD',repo/'outputs/ruod-v8-workload')
    with zipfile.ZipFile(repo/'colab/ULGF_checkpoint02_v8_refresh.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('refresh_ruod_v7.py','apply_ruod_visual_families.py','screen_ruod_near_duplicates.py','prepare_ruod_split.py','audit_ruod_dataset.py'):
            z.write(repo/'tools'/name,name)
        for name in ('spec.json',)+tuple(spec['payload_hashes']):
            z.write(bundle/name,name)

if __name__=='__main__':
    main()
