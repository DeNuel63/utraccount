"""Full v6 workload plus candidates connected to historical reviewed families."""
from pathlib import Path
from .assess_ruod_v5_workload import main as assess, key, groups
from .apply_ruod_visual_families import read, write
from .review_ruod_v3_candidates import compose_actions
from .screen_ruod_near_duplicates import screen_pairs

def main():
    assess('v6')
    repo=Path(__file__).resolve().parents[1]
    out=repo/'outputs/ruod-v6-workload'
    folders=['ruod-visual-family-proposal']+['ruod-visual-family-proposal-batch'+s for s in ('02','03','04','05')]
    manifests=[read(repo/'outputs'/f/'reviewed_visual_families.json') for f in folders]
    actions={a['member']:a['to_split'] for a in compose_actions([read(repo/'outputs'/f/'proposed_actions.json') for f in folders])}
    cached=read(repo/'outputs/ruod-near-duplicate-screen/fingerprints.json')
    all_rows={key(r):dict(r,split=actions.get(key(r),r['split'])) for r in cached}
    edges=[]
    for m in manifests:
        for e in m['reviewed_edges']:
            a,b=(key(m['members'][e[s]]) for s in ('left','right'))
            edges.append(dict(left=all_rows[a],right=all_rows[b],score=0))
    families=groups(edges)
    ownership={k:index for index,f in enumerate(families) for k in f['members']}
    # Search retained train/validation against ALL reviewed anchors, even removed ones.
    left=[r for r in all_rows.values() if r['split'] in ('train','validation')]
    anchors=[all_rows[k] for k in ownership]
    candidates,total=screen_pairs(left,anchors,keep=100000)
    if total!=len(candidates):
        raise ValueError('Historical-family candidate truncation')
    connections=[]
    seen=set()
    for e in candidates:
        a,b=key(e['left']),key(e['right'])
        if a==b or (a in ownership and ownership[a]==ownership[b]):
            continue
        pair=frozenset((a,b))
        if pair in seen:
            continue
        seen.add(pair)
        family=families[ownership[b]]
        connections.append(dict(e,reviewed_family_index=ownership[b],
            reviewed_family_touches_test=family['touches_test'],
            anchor_is_excluded=e['right']['split']=='exclude'))
    current_edges=sum([read(out/(b+'.json')) for b in ('train__test','validation__test','train__validation')],[])
    current_pairs={frozenset((key(e['left']),key(e['right']))) for e in current_edges}
    strong=[e for e in connections if e['score']<=4]
    analysis=dict(reviewed_family_components=len(families),reviewed_anchors=len(anchors),
        surviving_reviewed_cross_split_families=sum(len({all_rows[k]['split'] for k in f['members']} - {'exclude'})>1 for f in families),
        candidate_connections=len(connections),priority_connections=len(strong),
        priority_live_images=len({key(e['left']) for e in strong}),
        priority_with_excluded_anchor=sum(e['anchor_is_excluded'] for e in strong),
        priority_outside_current_cross_split_pairs=sum(frozenset((key(e['left']),key(e['right']))) not in current_pairs for e in strong),
        priority_to_test_linked_family=sum(e['reviewed_family_touches_test'] for e in strong),
        membership_changed=False,
        limitation='New connections are unreviewed candidates, not approvals. Historical excluded anchors are evidence only, never reinstated. Cached signatures reused.')
    write(out/'historical_reviewed_families.json',families)
    write(out/'historical_family_candidates.json',connections)
    write(out/'historical_family_summary.json',analysis)
    print('Historical connections:',analysis)

if __name__=='__main__':
    main()
