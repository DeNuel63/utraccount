"""Read-only CPU workload refresh from hash-bound cached signatures."""
import argparse
from collections import Counter
from pathlib import Path
try:
    from .apply_ruod_visual_families import read, write, contract, digest_file
    from .screen_ruod_near_duplicates import screen_pairs
except ImportError:
    from apply_ruod_visual_families import read, write, contract, digest_file
    from screen_ruod_near_duplicates import screen_pairs


def key(row):
    return row['origin']+':'+str(row['id'])


def components(edges):
    graph = {}
    for e in edges:
        a,b = key(e['left']),key(e['right'])
        graph.setdefault(a,set()).add(b)
        graph.setdefault(b,set()).add(a)
    result, seen = [],set()
    for node in sorted(graph):
        if node in seen:
            continue
        pending, group = [node],set()
        while pending:
            n = pending.pop()
            if n in group:
                continue
            group.add(n)
            pending.extend(graph[n]-group)
        seen.update(group)
        result.append(dict(members=sorted(group), touches_test=any(k.startswith('test:') for k in group)))
    return result


def refresh(base, bundle, source, output):
    if output.exists():
        raise ValueError('Use a fresh report directory')
    spec = read(bundle/'spec.json')
    for name,sha in spec['payload_hashes'].items():
        if digest_file(bundle/name) != sha:
            raise ValueError('Bundle hash mismatch: '+name)
    data = {s:read(base/'annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    for s,d in data.items():
        if contract(d) != spec['base_contracts'][s]:
            raise ValueError('Reviewed split contract mismatch: '+s)
    if digest_file(base/'annotations/instances_test.json') != spec['official_test_sha256']:
        raise ValueError('Official test bytes mismatch')
    for s,sha in spec['source_annotation_hashes'].items():
        if digest_file(source/'RUOD_ANN'/('instances_'+s+'.json')) != sha:
            raise ValueError('Original annotation hash mismatch: '+s)
    current = {('test' if s=='test' else 'train')+':'+str(i['id']):s for s,d in data.items() for i in d['images']}
    cached = read(bundle/'fingerprints.json')
    rows = {key(r):dict(r,split=current.get(key(r),'exclude')) for r in cached}
    if not set(current) <= set(rows):
        raise ValueError('Missing current fingerprints')
    for r in rows.values():
        r['path'] = str(source/'RUOD_pic'/r['origin']/r['file_name'])
    reviewed = read(bundle/'reviewed_edges.json')
    edges = [dict(left=rows[e['left']],right=rows[e['right']],score=0) for e in reviewed]
    families = components(edges)
    ownership = {k:index for index,g in enumerate(families) for k in g['members']}
    violations = [g for g in families if len({current[k] for k in g['members'] if k in current})>1]
    if violations:
        raise ValueError('Reviewed transitive family crosses current split')
    boundaries,all_edges = {},[]
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        left = [r for r in rows.values() if r['split']==a]
        right = [r for r in rows.values() if r['split']==b]
        found,total = screen_pairs(left,right,keep=100000)
        if len(found)!=total:
            raise ValueError('Candidate truncation')
        name = a+'__'+b
        boundaries[name] = dict(pairs_screened=len(left)*len(right),candidates=total,priority_pairs=sum(e['score']<=4 for e in found))
        write(output/(name+'.json'),found)
        all_edges.extend(found)
        print(name,boundaries[name],flush=True)
    live = [r for r in rows.values() if r['split'] in ('train','validation')]
    anchors = [rows[k] for k in ownership]
    found,total = screen_pairs(live,anchors,keep=100000)
    if len(found)!=total:
        raise ValueError('Historical candidate truncation')
    historical,seen = [],set()
    for e in found:
        a,b = key(e['left']),key(e['right'])
        pair = frozenset((a,b))
        if a==b or (a in ownership and ownership[a]==ownership[b]) or pair in seen:
            continue
        seen.add(pair)
        historical.append(dict(e,reviewed_family_index=ownership[b],
            reviewed_family_touches_test=families[ownership[b]]['touches_test'],
            anchor_is_excluded=b not in current))
    current_pairs = {frozenset((key(e['left']),key(e['right']))) for e in all_edges}
    strong = [e for e in all_edges if e['score']<=4]
    hstrong = [e for e in historical if e['score']<=4]
    actionable=[]
    for e in historical:
        family=families[e['reviewed_family_index']]
        partitions={current[k] for k in family['members'] if k in current}
        if family['touches_test'] or (partitions-{e['left']['split']}):
            actionable.append(e)
    priority_by_pair={frozenset((key(e['left']),key(e['right']))):dict(e,review_source='current_cross_split') for e in strong}
    for e in actionable:
        if e['score']<=4:
            pair=frozenset((key(e['left']),key(e['right'])))
            if pair not in priority_by_pair:
                priority_by_pair[pair]=dict(e,review_source='historical_family_connection')
    priority=sorted(priority_by_pair.values(),key=lambda e:(e['score'],key(e['left']),key(e['right'])))
    original = read(source/'RUOD_ANN/instances_train.json')
    retained = {i['id'] for s in ('train','validation') for i in data[s]['images']}
    exclusions = read(bundle/'exact_exclusions.json')
    reasons = {r['id']:r['reason'] for r in exclusions}
    original_ids = {i['id'] for i in original['images']}
    excluded = original_ids-retained
    if set(reasons) & retained or not set(reasons)<=excluded:
        raise ValueError('Exact exclusions inconsistent with retained membership')
    for i in excluded-set(reasons):
        reasons[i]='reviewed_visual_family'
    def distribution(dataset):
        count = Counter(a['category_id'] for a in dataset['annotations'])
        return {c['name']:count[c['id']] for c in original['categories']}
    coverage = dict(original_training_images=len(original_ids),retained_train_validation_images=len(retained),
        excluded_images=len(excluded),exclusion_reasons=dict(Counter(reasons.values())),
        annotations=dict(original_train=distribution(original),**{s:distribution(data[s]) for s in ('train','validation')}),
        excluded_annotations_by_reason={},class_retention={},scene_coverage='NOT_ESTABLISHED_BY_CLASS_COUNTS',
        conflicting_annotation_review='PENDING; excluded labels are not corrected or reintroduced')
    for reason in sorted(set(reasons.values())):
        coverage['excluded_annotations_by_reason'][reason]=distribution(dict(annotations=[a for a in original['annotations'] if reasons.get(a['image_id'])==reason]))
    for c in original['categories']:
        name=c['name']; start=coverage['annotations']['original_train'][name]
        remaining=sum(coverage['annotations'][s][name] for s in ('train','validation'))
        coverage['class_retention'][name]=dict(original_annotations=start,retained_annotations=remaining,
            retention_percent=round(100*remaining/start,2),excluded_annotations=start-remaining)
    report = dict(status='WORKLOAD_REFRESH_COMPLETE_REVIEW_PENDING',split=spec.get('split','v7-reviewed'),
        images={s:len(d['images']) for s,d in data.items()},base_contracts=spec['base_contracts'],
        boundaries=boundaries,candidate_pairs=len(all_edges),candidate_components=len(components(all_edges)),
        priority_pairs=len(strong),priority_components=len(components(strong)),
        historical=dict(reviewed_families=len(families),reviewed_anchors=len(ownership),
            confirmed_cross_split_families=0,candidate_connections=len(historical),priority_connections=len(hstrong),
            priority_live_images=len({key(e['left']) for e in hstrong}),
            actionable_connections=len(actionable),actionable_priority_connections=sum(e['score']<=4 for e in actionable),
            same_partition_or_no_retained_conflict_connections=len(historical)-len(actionable),
            priority_excluded_anchor_links=sum(e['anchor_is_excluded'] for e in hstrong),
            priority_outside_current_pairs=sum(frozenset((key(e['left']),key(e['right']))) not in current_pairs for e in hstrong)),
        membership_changed=False,training_started=False,training_ready=False,
        unique_actionable_priority_pairs=len(priority),unique_actionable_priority_components=len(components(priority)),
        limitation='Cached audited signatures, not fresh file/decode audit. Scores and components are NOT duplicate verdicts. Crops/mirrors may be missed. Coverage counts do not establish scene diversity.')
    write(output/'historical_family_candidates.json',historical)
    write(output/'historical_reviewed_families.json',families)
    write(output/'priority_components.json',components(strong))
    write(output/'actionable_priority_pairs.json',priority)
    write(output/'actionable_priority_components.json',components(priority))
    write(output/'exclusion_coverage.json',coverage)
    write(output/'summary.json',report)
    print(report,flush=True)
    print('Exclusion coverage:',coverage,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','bundle','source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); refresh(a.base,a.bundle,a.source,a.output)
