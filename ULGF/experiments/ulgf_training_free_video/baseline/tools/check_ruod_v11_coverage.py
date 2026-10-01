"""Read-only exact-leakage, exclusion reconciliation and class coverage checks."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
try:
    from .apply_ruod_visual_families import read,write,contract,digest_file
except ImportError:
    from apply_ruod_visual_families import read,write,contract,digest_file

def check(base,bundle,source,output):
    spec=read(bundle/'spec.json')
    for name,sha in spec['payload_hashes'].items():
        if digest_file(bundle/name)!=sha: raise ValueError('Changed evidence')
    data={s:read(base/'annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    for s,d in data.items():
        if contract(d)!=spec['base_contracts'][s]: raise ValueError('Changed split')
    original={s:read(source/'RUOD_ANN'/('instances_'+s+'.json')) for s in ('train','test')}
    for s in original:
        if digest_file(source/'RUOD_ANN'/('instances_'+s+'.json'))!=spec['source_annotation_hashes'][s]: raise ValueError('Changed original annotations')
    hashes=read(bundle/'exact_pixel_hashes.json'); ownership=defaultdict(set)
    integrity={}; classes={}; ids={}
    for s,d in data.items():
        ids[s]={i['id'] for i in d['images']}
        annids={a['id'] for a in d['annotations']}; catids={c['id'] for c in d['categories']}
        if len(ids[s])!=len(d['images']) or len(annids)!=len(d['annotations']): raise ValueError('Duplicate IDs')
        if any(a['image_id'] not in ids[s] or a['category_id'] not in catids for a in d['annotations']): raise ValueError('Orphan annotation')
        if d['categories']!=data['test']['categories']: raise ValueError('Category mapping mismatch')
        for i in d['images']: ownership[hashes[('test' if s=='test' else 'train')+':'+str(i['id'])]].add(s)
        classes[s]={c['name']:dict(images=len({a['image_id'] for a in d['annotations'] if a['category_id']==c['id']}),
            annotations=sum(a['category_id']==c['id'] for a in d['annotations'])) for c in d['categories']}
        integrity[s]='PASS'
    leaks=[sorted(v) for v in ownership.values() if len(v)>1]
    if leaks or ids['train'] & ids['validation']: raise ValueError('Exact leakage or overlapping training/validation identities')
    retained=ids['train']|ids['validation']; original_ids={i['id'] for i in original['train']['images']}
    excluded=original_ids-retained
    if not retained<=original_ids: raise ValueError('Unknown training identities')
    originals={a['id']:a for a in original['train']['annotations']}
    kept=[a for s in ('train','validation') for a in data[s]['annotations']]
    expected={a['id'] for a in originals.values() if a['image_id'] in retained}
    if {a['id'] for a in kept}!=expected: raise ValueError('Annotation accounting mismatch')
    for a in kept:
        if any(a[k]!=originals[a['id']][k] for k in ('image_id','category_id','bbox')): raise ValueError('Unexpected label/box change')
    reasons={r['id']:r['reason'] for r in read(bundle/'exact_exclusions.json')}
    if set(reasons)&retained or not set(reasons)<=excluded: raise ValueError('Excluded record returned')
    accounted=Counter(reasons.get(i,'reviewed_visual_family') for i in excluded)
    report=dict(status='MECHANICAL_COVERAGE_AND_EXACT_LEAKAGE_PASS',split=spec['split'],
        annotation_integrity=integrity,exact_cross_split_pixel_leakage_groups=0,
        training_validation_image_overlap=0,excluded_images=len(excluded),exclusion_reasons=dict(accounted),
        retained_annotation_accounting='PASS',retained_categories_and_boxes_unchanged='PASS',class_coverage=classes,
        all_classes_present=all(v['images']>0 for d in classes.values() for v in d.values()),
        semantic_label_quality='NOT_ESTABLISHED',scene_diversity='NOT_ESTABLISHED',
        conflict_label_recovery='NOT_PERFORMED; conflicting originals remain excluded',
        membership_changed=False,training_ready=False,
        limitation='Uses audited cached pixel hashes; not a fresh decode audit. Mechanical accounting cannot certify visual diversity, label correctness or near-duplicate clearance.')
    write(output/'coverage_checks.json',report); print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','bundle','source','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); check(a.base,a.bundle,a.source,a.output)
