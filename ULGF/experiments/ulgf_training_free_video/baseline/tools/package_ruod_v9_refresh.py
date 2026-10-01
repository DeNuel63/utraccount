"""Reconstruct Colab-confirmed v9 and refresh candidates with all reviewed edges."""
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, contract, digest_file
from .apply_ruod_consolidated import revise
from .refresh_ruod_v7 import refresh

def main():
    repo=Path(__file__).resolve().parents[1]
    old=repo/'outputs/ruod-v8-refresh-inputs'
    base=repo/'outputs/ruod-v9-refresh-base'
    bundle=repo/'outputs/ruod-v9-refresh-inputs'
    mp=repo/'outputs/ruod-v9-consolidated-proposal/reviewed_manifest.json'
    if digest_file(mp)!='1ed3ca25918cd8c4d0485b918506da31ed6d94b9c8fa62c05db9dfefa724ef64':
        raise ValueError('Manifest differs from Colab-confirmed application')
    manifest=read(mp)
    data={s:read(repo/'outputs/ruod-v8-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    data,actions=revise(data,manifest)
    for s,d in data.items(): write(base/'annotations'/('instances_'+s+'.json'),d)
    for name in ('fingerprints.json','exact_exclusions.json'): write(bundle/name,read(old/name))
    edges=read(old/'reviewed_edges.json')+manifest['reviewed_edges']
    edges=[dict(left=a,right=b) for a,b in sorted({tuple(sorted((e['left'],e['right']))) for e in edges})]
    write(bundle/'reviewed_edges.json',edges)
    spec=dict(read(old/'spec.json'),split='v9-reviewed',base_contracts={s:contract(d) for s,d in data.items()},
        applied_manifest_sha256=digest_file(mp),
        payload_hashes={n:digest_file(bundle/n) for n in ('fingerprints.json','exact_exclusions.json','reviewed_edges.json')})
    write(bundle/'spec.json',spec)
    local=repo/'outputs/ruod-v9-local-refresh-inputs'
    for name in spec['payload_hashes']: write(local/name,read(bundle/name))
    write(local/'spec.json',dict(spec,official_test_sha256=digest_file(base/'annotations/instances_test.json')))
    refresh(base,local,repo.parent/'RUOD',repo/'outputs/ruod-v9-workload')
    archive=repo/'colab/ULGF_checkpoint02_v9_refresh.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('refresh_ruod_v7.py','apply_ruod_visual_families.py','screen_ruod_near_duplicates.py','prepare_ruod_split.py','audit_ruod_dataset.py'):
            z.write(repo/'tools'/name,name)
        for name in ('spec.json',)+tuple(spec['payload_hashes']): z.write(bundle/name,name)
    print('ZIP SHA256:',digest_file(archive))

if __name__=='__main__': main()
