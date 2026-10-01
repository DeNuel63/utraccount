"""Count remaining v5 screening workload; no visual decisions or split edits."""
from collections import Counter
from pathlib import Path
from .apply_ruod_visual_families import read, write, revise, digest_file, contract
from .build_ruod_visual_manifest import reconstruct_base
from .review_ruod_v3_candidates import compose_actions
from .screen_ruod_near_duplicates import screen_pairs


def key(row):
    return row['origin'] + ':' + str(row['id'])


def groups(edges):
    parent = {}
    def find(k):
        parent.setdefault(k,k)
        if parent[k] != k:
            parent[k] = find(parent[k])
        return parent[k]
    for edge in edges:
        a,b = key(edge['left']),key(edge['right'])
        parent[find(a)] = find(b)
    components = {}
    for k in list(parent):
        components.setdefault(find(k),set()).add(k)
    rows = []
    for members in components.values():
        links = [e for e in edges if key(e['left']) in members]
        rows.append(dict(members=sorted(members), member_count=len(members), links=len(links),
            best_score=min(e['score'] for e in links), touches_test=any(k.startswith('test:') for k in members)))
    return sorted(rows,key=lambda r:(r['best_score'],-r['links'],r['members']))


def main(version='v5'):
    repo = Path(__file__).resolve().parents[1]
    screen = repo/'outputs/ruod-near-duplicate-screen'
    if version not in ('v5','v6'):
        raise ValueError('Unsupported split version')
    output = repo/('outputs/ruod-'+version+'-workload')
    batches=('02','03','04') if version=='v5' else ('02','03','04','05')
    folders = ['ruod-visual-family-proposal'] + ['ruod-visual-family-proposal-batch'+s for s in batches]
    manifests = [read(repo/'outputs'/f/'reviewed_visual_families.json') for f in folders]
    base = reconstruct_base(repo.parent/'RUOD',repo/'outputs/ruod-local-audit')
    for manifest in manifests:
        base = revise(base,manifest)[0]
    current = {('test' if s=='test' else 'train')+':'+str(i['id']):s
               for s,d in base.items() for i in d['images']}
    expected=dict(train=7719,validation=868,test=4200) if version=='v5' else dict(train=7692,validation=852,test=4200)
    assert Counter(current.values()) == expected
    for manifest in manifests:
        for edge in manifest['reviewed_edges']:
            a,b = (key(manifest['members'][edge[side]]) for side in ('left','right'))
            if a in current and b in current and current[a] != current[b]:
                raise ValueError('Previously reviewed edge still crosses splits')
    cached = read(screen/'fingerprints.json')
    lookup = {key(r):dict(r,split=current[key(r)]) for r in cached if key(r) in current}
    assert set(lookup) == set(current)
    # Bind cached signatures to the audited source identities without decoding images again.
    for origin in ('train','test'):
        audit = {r['id']:r for r in read(repo/('outputs/ruod-local-audit/audit/'+origin+'.json'))}
        for r in lookup.values():
            if r['origin']==origin and r['source_sha256'] != audit[r['id']]['file_sha256']:
                raise ValueError('Fingerprint source identity mismatch')
    saved_edges, saved_stats = [],Counter()
    for boundary in ('train__validation','train__test','validation__test'):
        for e in read(screen/(boundary+'.json')):
            a,b = key(e['left']),key(e['right'])
            if a not in current or b not in current:
                saved_stats['excluded_endpoint'] += 1
            elif current[a]==current[b]:
                saved_stats['now_same_partition'] += 1
            else:
                saved_stats['still_cross_split'] += 1
                saved_edges.append(dict(e,left=lookup[a],right=lookup[b]))
    all_edges,boundaries = [],{}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        left=[r for r in lookup.values() if r['split']==a]
        right=[r for r in lookup.values() if r['split']==b]
        edges,total=screen_pairs(left,right,keep=100000)
        if len(edges)!=total:
            raise ValueError('Truncated workload; increase bound')
        name=a+'__'+b
        boundaries[name]=dict(pairs_screened=len(left)*len(right),threshold_candidates=total,
            strongest_score_zero=sum(e['score']==0 for e in edges),
            score_at_most_4=sum(e['score']<=4 for e in edges))
        write(output/(name+'.json'),edges)
        all_edges.extend(edges)
        print(name,boundaries[name],flush=True)
    components=groups(all_edges)
    strong_edges=[e for e in all_edges if e['score']<=4]
    strong_components=groups(strong_edges)
    shortlist_pairs={frozenset((key(e['left']),key(e['right']))) for e in saved_edges}
    report=dict(current_split=version+'-reviewed',images=dict(Counter(current.values())),
        base_contracts={s:contract(d) for s,d in base.items()},
        fingerprint_cache_sha256=digest_file(screen/'fingerprints.json'),
        method='Recompare all current cross-split pairs using cached audited pHash/dHash; pHash<=8 OR dHash<=6',
        original_saved_shortlist=dict(saved_stats),remaining_saved_shortlist_components=len(groups(saved_edges)),
        boundaries=boundaries,total_remaining_candidate_pairs=len(all_edges),
        outside_original_saved_shortlist=sum(frozenset((key(e['left']),key(e['right']))) not in shortlist_pairs for e in all_edges),
        candidate_components=len(components),candidate_images=len({key(e[k]) for e in all_edges for k in ('left','right')}),
        previous_reviewed_cross_split_edges_remaining=0,
        priority_score_at_most_4=dict(pairs=len(strong_edges),components=len(strong_components),
            images=len({key(e[k]) for e in strong_edges for k in ('left','right')}),
            components_touching_test=sum(c['touches_test'] for c in strong_components)),
        components_touching_test=sum(c['touches_test'] for c in components),
        training_validation_only_components=sum(not c['touches_test'] for c in components),
        largest_component_images=max(c['member_count'] for c in components),
        strongest_components=components[:15],membership_changed=False,training_ready=False,
        limitation='Candidate components are NOT verified visual families. Weak links may join unrelated scenes. Cached signatures reused; no fresh image-file hash or decode audit. Crops/mirrors can escape thresholds; counts are not full leakage clearance.')
    write(output/'candidate_components.json',components)
    write(output/'summary.json',report)
    print({k:v for k,v in report.items() if k!='strongest_components'})

if __name__=='__main__':
    main()
