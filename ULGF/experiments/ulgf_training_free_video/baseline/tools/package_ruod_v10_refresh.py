"""Read-only v10 refresh bound to the confirmed Colab application."""
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, contract, digest_file
from .apply_ruod_consolidated import revise
from .refresh_ruod_v7 import refresh

def main():
    repo=Path(__file__).resolve().parents[1]
    old=repo/'outputs/ruod-v9-refresh-inputs'
    base=repo/'outputs/ruod-v10-refresh-base'
    bundle=repo/'outputs/ruod-v10-refresh-inputs'
    mp=repo/'outputs/ruod-v10-historical10-proposal/reviewed_manifest.json'
    confirmed='5a88968f29e91488ea53901073da34612d1aa0746cffeca48e7c06b41745b8f5'
    if digest_file(mp)!=confirmed: raise ValueError('Manifest differs from Colab confirmation')
    manifest=read(mp)
    data={s:read(repo/'outputs/ruod-v9-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    data,actions=revise(data,manifest)
    if len(actions)!=10: raise ValueError('Unexpected scope')
    for s,d in data.items(): write(base/'annotations'/('instances_'+s+'.json'),d)
    for name in ('fingerprints.json','exact_exclusions.json'): write(bundle/name,read(old/name))
    edges=read(old/'reviewed_edges.json')+manifest['reviewed_edges']
    edges=[dict(left=a,right=b) for a,b in sorted({tuple(sorted((e['left'],e['right']))) for e in edges})]
    write(bundle/'reviewed_edges.json',edges)
    spec=dict(read(old/'spec.json'),split='v10-reviewed',base_contracts={s:contract(d) for s,d in data.items()},
        applied_manifest_sha256=confirmed,
        payload_hashes={n:digest_file(bundle/n) for n in ('fingerprints.json','exact_exclusions.json','reviewed_edges.json')})
    write(bundle/'spec.json',spec)
    local=repo/'outputs/ruod-v10-local-refresh-inputs'
    for name in spec['payload_hashes']: write(local/name,read(bundle/name))
    write(local/'spec.json',dict(spec,official_test_sha256=digest_file(base/'annotations/instances_test.json')))
    refresh(base,local,repo.parent/'RUOD',repo/'outputs/ruod-v10-workload')
    archive=repo/'colab/ULGF_checkpoint02_v10_refresh.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('refresh_ruod_v7.py','apply_ruod_visual_families.py','screen_ruod_near_duplicates.py','prepare_ruod_split.py','audit_ruod_dataset.py'):
            z.write(repo/'tools'/name,name)
        for name in ('spec.json',)+tuple(spec['payload_hashes']): z.write(bundle/name,name)
    cell=(repo/'colab/ULGF_checkpoint02_v9_refresh_cell.py').read_text(encoding='utf-8')
    cell=cell.replace('v9','v10').replace('ruod-visual-families-v10-consolidated','ruod-visual-families-v10-historical10')
    cell=cell.replace('1ed3ca25918cd8c4d0485b918506da31ed6d94b9c8fa62c05db9dfefa724ef64',confirmed)
    cell=cell.replace('208585230306dbda07c6f2e03feb27daa431d8a24a30a2852dec8fc09672c0c9',digest_file(archive))
    (repo/'colab/ULGF_checkpoint02_v10_refresh_cell.py').write_text(cell,encoding='utf-8')
    print('ZIP SHA256:',digest_file(archive))

if __name__=='__main__': main()
