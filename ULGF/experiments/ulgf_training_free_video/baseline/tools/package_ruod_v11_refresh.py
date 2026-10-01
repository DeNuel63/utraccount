"""Read-only v11 refresh and coverage audit, bound to Colab confirmation."""
from pathlib import Path
import zipfile
from .apply_ruod_visual_families import read, write, contract, digest_file
from .apply_ruod_consolidated import revise
from .refresh_ruod_v7 import refresh
from .check_ruod_v11_coverage import check

def main():
    repo=Path(__file__).resolve().parents[1]; old=repo/'outputs/ruod-v10-refresh-inputs'
    base=repo/'outputs/ruod-v11-refresh-base'; bundle=repo/'outputs/ruod-v11-refresh-inputs'
    mp=repo/'outputs/ruod-v11-expanded30-proposal/reviewed_manifest.json'
    confirmed='47f1cdcc142b7cfd84e7dc15085956f6c317740b5b38d850a04d7baa03edb3a5'
    if digest_file(mp)!=confirmed: raise ValueError('Manifest differs from Colab confirmation')
    manifest=read(mp)
    data={s:read(repo/'outputs/ruod-v10-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    data,actions=revise(data,manifest)
    if len(actions)!=30: raise ValueError('Unexpected scope')
    for s,d in data.items(): write(base/'annotations'/('instances_'+s+'.json'),d)
    for name in ('fingerprints.json','exact_exclusions.json'): write(bundle/name,read(old/name))
    edges=read(old/'reviewed_edges.json')+manifest['reviewed_edges']
    edges=[dict(left=a,right=b) for a,b in sorted({tuple(sorted((e['left'],e['right']))) for e in edges})]
    write(bundle/'reviewed_edges.json',edges)
    # Audited exact pixel hashes augment perceptual signatures for a fresh
    # membership-level exact leakage check (not a new file decode audit).
    exact={}
    for origin in ('train','test'):
        for r in read(repo/'outputs/ruod-local-audit/audit'/(origin+'.json')):
            exact[origin+':'+str(r['id'])]=r['stored_pixels_sha256']
    write(bundle/'exact_pixel_hashes.json',exact)
    names=('fingerprints.json','exact_exclusions.json','reviewed_edges.json','exact_pixel_hashes.json')
    spec=dict(read(old/'spec.json'),split='v11-reviewed',base_contracts={s:contract(d) for s,d in data.items()},
        applied_manifest_sha256=confirmed,payload_hashes={n:digest_file(bundle/n) for n in names})
    write(bundle/'spec.json',spec)
    local=repo/'outputs/ruod-v11-local-refresh-inputs'
    for name in names: write(local/name,read(bundle/name))
    write(local/'spec.json',dict(spec,official_test_sha256=digest_file(base/'annotations/instances_test.json')))
    report=repo/'outputs/ruod-v11-workload'
    refresh(base,local,repo.parent/'RUOD',report)
    check(base,local,repo.parent/'RUOD',report)
    archive=repo/'colab/ULGF_checkpoint02_v11_refresh.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('refresh_ruod_v7.py','check_ruod_v11_coverage.py','apply_ruod_visual_families.py','screen_ruod_near_duplicates.py','prepare_ruod_split.py','audit_ruod_dataset.py'):
            z.write(repo/'tools'/name,name)
        for name in ('spec.json',)+names: z.write(bundle/name,name)
    cell=(repo/'colab/ULGF_checkpoint02_v10_refresh_cell.py').read_text(encoding='utf-8')
    cell=cell.replace('v10','v11').replace('ruod-visual-families-v11-historical10','ruod-visual-families-v11-expanded30')
    cell=cell.replace('5a88968f29e91488ea53901073da34612d1aa0746cffeca48e7c06b41745b8f5',confirmed)
    cell=cell.replace('66c9b9c88576e3ba2b9de3e0e2eabf82fc9839f5e71c0e9ef760b0abad7def76',digest_file(archive))
    cell=cell.replace("expected={'refresh_ruod_v7.py'", "expected={'check_ruod_v11_coverage.py','exact_pixel_hashes.json','refresh_ruod_v7.py'")
    cell=cell.replace("print('Reports:',reports/'assessment')", "coverage_command=[sys.executable,'-u',str(workspace/'check_ruod_v11_coverage.py')]+command[3:]\ncoverage=subprocess.run(coverage_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\nprint(coverage.stdout)\n(reports/'coverage.log').write_text(coverage.stdout)\ncoverage.check_returncode()\nprint('Reports:',reports/'assessment')")
    (repo/'colab/ULGF_checkpoint02_v11_refresh_cell.py').write_text(cell,encoding='utf-8')
    print('ZIP SHA256:',digest_file(archive))

if __name__=='__main__': main()
