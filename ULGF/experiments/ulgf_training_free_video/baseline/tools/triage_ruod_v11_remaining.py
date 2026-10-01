"""Render a bounded diagnostic sample of strongest remaining weak candidates."""
from pathlib import Path
from collections import Counter
from .apply_ruod_visual_families import read,write,digest_file
from .screen_ruod_near_duplicates import contact_sheet

def main():
    repo=Path(__file__).resolve().parents[1]; root=repo/'outputs/ruod-v11-workload'
    out=repo/'outputs/ruod-v11-weak-triage'; out.mkdir(exist_ok=True)
    edges=[]
    for name in ('train__validation','train__test','validation__test'): edges.extend(read(root/(name+'.json')))
    edges=sorted(edges,key=lambda e:(e['score'],e['left']['origin'],e['left']['id'],e['right']['id']))
    chosen=[]; seen=set()
    for e in edges:
        k=(e['left']['origin'],e['left']['id'])
        if k not in seen: chosen.append(e); seen.add(k)
        if len(chosen)==12: break
    members={r['origin']+':'+str(r['id']):r for e in chosen for r in (e['left'],e['right'])}
    for r in members.values():
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Changed original')
    for i in range(0,len(chosen),4): contact_sheet(chosen,out/('pairs_%02d.png'%(i//4+1)),i,4)
    write(out/'inventory.json',dict(pairs=chosen,members=members,hashes_checked=len(members),
        score_distribution=dict(Counter(e['score'] for e in edges)),sample_method='Twelve strongest direct flags with distinct left source identities; diagnostic, not representative.',membership_changed=False))
    print('Sampled:',len(chosen),'hashes:',len(members))

if __name__=='__main__': main()
