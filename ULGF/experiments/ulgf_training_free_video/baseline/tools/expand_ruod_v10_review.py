"""Find the next unreviewed boundary around manually accepted family members."""
from pathlib import Path
from .apply_ruod_visual_families import read,write,digest_file
from .screen_ruod_near_duplicates import screen_pairs,contact_sheet

def main():
    repo=Path(__file__).resolve().parents[1]; out=repo/'outputs/ruod-v10-expanded-review'
    inventory=read(out/'inventory.json'); key=lambda r:r['origin']+':'+str(r['id'])
    accepted=set(inventory['members'])
    # Added only after their contact sheets have been manually inspected.
    decisions=out/'additional_accepted.json'
    if decisions.exists(): accepted.update(read(decisions)['accepted_keys'])
    data={s:read(repo/'outputs/ruod-v10-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    current={('test' if s=='test' else 'train')+':'+str(i['id']):s for s,d in data.items() for i in d['images']}
    rows={key(r):dict(r,split=current.get(key(r),'exclude')) for r in read(repo/'outputs/ruod-v10-refresh-inputs/fingerprints.json')}
    left=[r for k,r in rows.items() if k in current and k not in accepted]
    found,total=screen_pairs(left,[rows[k] for k in sorted(accepted)],keep=100000)
    if len(found)!=total: raise ValueError('Truncated candidates')
    best={}
    for e in found: best.setdefault(key(e['left']),e)
    pairs=list(best.values()); round_id=1 if not decisions.exists() else read(decisions)['next_round']
    members={key(r):r for e in pairs for r in (e['left'],e['right'])}
    for r in members.values():
        r['path']=str(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Changed source')
    for start in range(0,len(pairs),4): contact_sheet(pairs,out/('round%d_%02d.png'%(round_id,start//4+1)),start,4)
    write(out/('round%d.json'%round_id),dict(pairs=pairs,members=members,candidate_links=total,accepted_anchor_count=len(accepted),
        threshold=dict(phash_max=8,dhash_max=6,rule='OR'),membership_changed=False))
    print('Round',round_id,'new images',len(pairs),'links',total)
    print([(e['left']['file_name'],e['left']['split'],e['right']['file_name'],e['score']) for e in pairs])

if __name__=='__main__': main()
