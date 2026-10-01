"""Build source-bound v7 workload inputs and run the local refresh."""
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, revise, contract, digest_file
from .apply_ruod_historical22 import revise as revise22
from .build_ruod_visual_manifest import reconstruct_base
from .prepare_ruod_split import partition
from .refresh_ruod_v7 import refresh


def main():
    repo=Path(__file__).resolve().parents[1]; source=repo.parent/'RUOD'
    bundle=repo/'outputs/ruod-v7-refresh-inputs'
    base=repo/'outputs/ruod-v7-refresh-base'
    data=reconstruct_base(source,repo/'outputs/ruod-local-audit')
    folders=['ruod-visual-family-proposal']+['ruod-visual-family-proposal-batch'+s for s in ('02','03','04','05')]
    edges=[]
    for f in folders:
        m=read(repo/'outputs'/f/'reviewed_visual_families.json')
        data=revise(data,m)[0]
        for e in m['reviewed_edges']:
            ends=[m['members'][e[s]] for s in ('left','right')]
            edges.append(dict(left=ends[0]['origin']+':'+str(ends[0]['id']),right=ends[1]['origin']+':'+str(ends[1]['id'])))
    latest=read(repo/'outputs/ruod-historical22-proposal/reviewed_manifest.json')
    data=revise22(data,latest)
    edges.extend(dict(left=e['left'],right=e['right']) for e in latest['reviewed_edges'])
    edges=[dict(left=a,right=b) for a,b in sorted({tuple(sorted((e['left'],e['right']))) for e in edges})]
    records={s:read(repo/'outputs/ruod-local-audit/audit'/(s+'.json')) for s in ('train','test')}
    originals={s:read(source/'RUOD_ANN'/('instances_'+s+'.json')) for s in records}
    for s,d in originals.items():
        lookup={r['id']:r for r in records[s]}
        d['images']=[dict(i,width=lookup[i['id']]['stored_size'][0],height=lookup[i['id']]['stored_size'][1]) for i in d['images']]
    exclusions=partition(originals['train'],originals['test'],records)[3]
    fingerprints=read(repo/'outputs/ruod-near-duplicate-screen/fingerprints.json')
    for r in fingerprints:
        record=next(x for x in records[r['origin']] if x['id']==r['id'])
        if r['source_sha256']!=record['file_sha256']:
            raise ValueError('Fingerprint/audit source mismatch')
    write(bundle/'fingerprints.json',fingerprints)
    write(bundle/'reviewed_edges.json',edges)
    write(bundle/'exact_exclusions.json',exclusions)
    spec=dict(base_contracts={s:contract(d) for s,d in data.items()},
        source_annotation_hashes=latest['source_annotation_hashes'],
        official_test_sha256=latest['official_test_sha256'],
        payload_hashes={n:digest_file(bundle/n) for n in ('fingerprints.json','reviewed_edges.json','exact_exclusions.json')})
    write(bundle/'spec.json',spec)
    for s,d in data.items():
        write(base/'annotations'/('instances_'+s+'.json'),d)
    # Local reconstructed JSON may differ in whitespace; bind the local-only run
    # to its bytes while the distributed specification preserves Colab test SHA.
    local_spec=dict(spec,official_test_sha256=digest_file(base/'annotations/instances_test.json'))
    local_bundle=repo/'outputs/ruod-v7-local-refresh-inputs'
    for name in spec['payload_hashes']:
        write(local_bundle/name,read(bundle/name))
    write(local_bundle/'spec.json',local_spec)
    output=repo/'outputs/ruod-v7-workload'
    refresh(base,local_bundle,source,output)
    names=('refresh_ruod_v7.py','apply_ruod_visual_families.py','screen_ruod_near_duplicates.py','prepare_ruod_split.py','audit_ruod_dataset.py')
    with zipfile.ZipFile(repo/'colab/ULGF_checkpoint02.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(repo/'tools'/name,name)
        for name in ('spec.json',)+tuple(spec['payload_hashes']):
            z.write(bundle/name,name)


if __name__=='__main__':
    main()
