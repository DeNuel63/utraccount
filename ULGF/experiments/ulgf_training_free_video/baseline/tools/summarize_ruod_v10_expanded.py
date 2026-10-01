"""Validate expanded visual decisions and summarize exact proposed actions."""
from pathlib import Path
from collections import Counter
from .apply_ruod_visual_families import read,write,digest_file,contract
from .apply_ruod_consolidated import revise

def main():
    repo=Path(__file__).resolve().parents[1]; out=repo/'outputs/ruod-v10-expanded-review'
    review=read(out/'visual_review.json'); boundary=read(out/'round7.json')
    if boundary['pairs'] or boundary['candidate_links']: raise ValueError('Expansion incomplete')
    for name,sha in review['evidence_hashes'].items():
        if digest_file(out/name)!=sha: raise ValueError('Changed review inputs')
    prior_path=repo/'outputs/ruod-v10-refresh-inputs/reviewed_edges.json'
    if digest_file(prior_path)!=review['prior_edges_sha256']: raise ValueError('Changed prior evidence')
    data={s:read(repo/'outputs/ruod-v10-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    current={('test' if s=='test' else 'train')+':'+str(i['id']):s for s,d in data.items() for i in d['images']}
    # Explicit cross-sheet comparison: the two initially separate anchor groups
    # converge on the same shell/urchin layout in round6 pairs 1 and 3.
    bridge=dict(left='train:7397',right='train:353',review_status='visually_reviewed',
        finding='Same bright upper shell, pink lower shell, upper-right urchin cluster and surrounding sea cucumbers; matching scene under slight framing change.',
        evidence_sheet='round6_01.png',evidence_pair_indices=[1,3],verdict='same_scene_family_supported')
    edges=read(prior_path)+review['decisions']+[bridge]
    graph={}
    for e in edges:
        a,b=e['left'],e['right']; graph.setdefault(a,set()).add(b); graph.setdefault(b,set()).add(a)
    actions=[]
    for k in sorted({d['left'] for d in review['decisions']}):
        if current.get(k) not in ('train','validation'): continue
        seen=set(); todo=[k]
        while todo:
            n=todo.pop()
            if n in seen: continue
            seen.add(n); todo.extend(graph.get(n,set())-seen)
        if not any(current.get(n)=='test' for n in seen): raise ValueError('Missing reviewed test path')
        actions.append(dict(member=k,file_name=review['members'][k]['file_name'],from_split=current[k],to_split='exclude'))
    counts=Counter(a['from_split'] for a in actions)
    expected={s:len(d['images'])-counts[s] for s,d in data.items()}
    lookup={r['origin']+':'+str(r['id']):r for r in read(repo/'outputs/ruod-v10-refresh-inputs/fingerprints.json')}
    manifest=dict(base_contracts={s:contract(d) for s,d in data.items()},members=lookup,
        reviewed_edges=[dict(e,review_status='visually_reviewed') for e in edges],proposed_actions=actions,expected_images=expected)
    revise(data,manifest)  # Exact full-history closure must match proposed actions.
    for r in review['members'].values():
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Changed source')
    summary=dict(status='EXPANDED_VISUAL_REVIEW_COMPLETE_APPLICATION_PENDING',base_split='v10-reviewed',
        reviewed_new_links=len(review['decisions']),explicit_cross_sheet_bridges=[bridge],
        original_hashes_checked=len(review['members']),proposed_actions=actions,exclusions_by_split=dict(counts),
        proposed_images=expected,moves=0,membership_changed=False,training_ready=False,
        expansion_rounds_reviewed=6,terminal_scan='round7.json',terminal_scan_sha256=digest_file(out/'round7.json'),
        remaining_boundary_candidates=0,limitation='Zero boundary links only for these expanded families under cached pHash<=8 OR dHash<=6 screening. Not global leakage clearance. Colab hash checks required before changes.')
    write(out/'review_summary.json',summary)
    print('Exclusions:',dict(counts),'expected:',expected,'hashes:',len(review['members']))
    print('Validation exclusions:',[a['file_name'] for a in actions if a['from_split']=='validation'])

if __name__=='__main__': main()
