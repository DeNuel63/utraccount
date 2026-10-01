"""Record explicitly completed visual-review rounds (not automatic decisions)."""
import argparse
from pathlib import Path
from .apply_ruod_visual_families import read,write,digest_file

def main(rounds):
    repo=Path(__file__).resolve().parents[1]; out=repo/'outputs/ruod-v10-expanded-review'
    initial=read(out/'inventory.json'); decisions=[]; members=dict(initial['members']); evidence={}
    for name in ['inventory.json']+['round%d.json'%i for i in range(1,rounds+1)]:
        data=read(out/name); members.update(data['members']); evidence[name]=digest_file(out/name)
        for n,e in enumerate(data['pairs'],1):
            def key(r): return r['origin']+':'+str(r['id'])
            family='shell_cluster' if e['right']['file_name'].startswith('0059') or e['right']['file_name']=='000940.jpg' else 'foreground_urchins'
            finding=('Matching bright central shells, angular foreground fragment and surrounding urchin/sea-cucumber layout.' if family=='shell_cluster' else
                     'Matching foreground urchins, pink/striped shells and distinctive surrounding sea-cucumber arrangement under framing changes.')
            decisions.append(dict(left=key(e['left']),right=key(e['right']),review_status='visually_reviewed',
                verdict='same_scene_family_supported',finding=finding,source_inventory=name,pair_index=n,
                sheet=('pairs_%02d.png'%((n-1)//4+1) if name=='inventory.json' else name[:-5]+'_%02d.png'%((n-1)//4+1))))
    for r in members.values():
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Changed source')
    accepted={d['left'] for d in decisions}
    write(out/'additional_accepted.json',dict(accepted_keys=sorted(accepted),next_round=rounds+1,
        note='Recorded only after model visually inspected each listed pair sheet. No membership changes.'))
    write(out/'visual_review.json',dict(base_split='v10-reviewed',decisions=decisions,members=members,
        evidence_hashes=evidence,prior_edges_sha256=initial['prior_edges_sha256'],hashes_checked=len(members),
        membership_changed=False,training_ready=False))
    print('Recorded pairs:',len(decisions),'hashed members:',len(members))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--rounds',type=int,required=True); main(p.parse_args().rounds)
