"""Record completed manual inspection of all 33 v8 priority contact sheets."""
from pathlib import Path
from collections import Counter
from .apply_ruod_visual_families import read, write, digest_file
from .refresh_ruod_v7 import components

FINDINGS=[
([1,2,11,17,34,38,47,49,54,55,61,66,71,74,79,87,93,94,103,104,109,117,118,126,129,130],
 'Matching shell, urchin and sea-cucumber arrangement on the seabed; small framing/blur differences.'),
([98,102,107,115,123], 'Matching sea-cucumber cluster, upper-right urchins and surrounding sandy seabed.'),
([3,13,26,30,40,45,46,48,60,62,64,72,113,122],
 'Same aquarium coral structures and rock arrangement; fish positions or crop differ.'),
([4,5,9,16,18,28,29,37,52,56,63,73,89,111,119],
 'Matching turtle appearance/pose and scene composition, with small framing, motion or watermark differences.'),
([6,7,15,19,41,43,50,59,78,90,97,101,105,114,116,121],
 'Matching diver equipment/positions and surrounding reef or animal composition; small pose/crop changes.'),
([8,10,12,14,22,25,27,32,53,57,58,67,68,70,80,83,85,86,88,96,99,125],
 'Matching cuttlefish pattern/placement and distinctive reef or rock background; small motion/crop differences.'),
([21,23,36,39,77,81,92,100,108,112,124],
 'Matching green-water seabed details, foreground equipment and shell/animal positions.'),
([31,35,44,69,84],
 'Matching rock contours, urchins/starfish and seabed marks; some pairs differ strongly in colour/exposure.'),
([33,65,76,91,95,120], 'Matching two lionfish and textured rock backdrop; small fin/pose changes.'),
([24,106,128], 'Matching squid positions and distinctive branching coral backdrop.'),
([42], 'Matching jellyfish bell lobes, trailing tentacles and green-water composition; slight movement.'),
([75], 'Matching jellyfish and surrounding butterflyfish over the same reef.'),
([110], 'Matching jellyfish bell markings and tentacle geometry against the same blue gradient.'),
([51], 'Matching cuttlefish on dark sandy bottom, red foreground object and red upper background landmark.'),
([20,82,127], 'Matching fish arrangement and identifiable coral/rock structure; minor motion differences.')]

def main():
    repo=Path(__file__).resolve().parents[1]; out=repo/'outputs/ruod-v8-priority-review'
    inventory=read(out/'inventory.json'); links=inventory['links']
    findings={i:text for ids,text in FINDINGS for i in ids}
    if set(findings)!=set(range(1,131)) or sum(len(ids) for ids,_ in FINDINGS)!=130:
        raise ValueError('Incomplete/duplicate visual findings')
    for r in inventory['members'].values():
        if digest_file(Path(r['path']))!=r['source_sha256']:
            raise ValueError('Source changed since inspection')
    def key(r): return r['origin']+':'+str(r['id'])
    decisions=[dict(pair_index=i,left=key(e['left']),right=key(e['right']),
        verdict='same_scene_family_supported',review_status='visually_reviewed',finding=findings[i],
        sheet='pairs_{:02d}.png'.format((i-1)//4+1)) for i,e in enumerate(links,1)]
    # Explicit additional comparisons observed across the already inspected
    # sheets. These are evidence-bearing bridges, not hash-driven expansion.
    bridges=[]
    for a,b,finding in [
        (3,13,'Same red plate coral, purple branches, yellow upper coral and lower honeycomb coral; wider view in pair 13.'),
        (27,8,'Same large left coral mound and central branching reef around the cuttlefish; changed tentacle pose.'),
        (74,17,'Same foreground three-urchin arrangement, shells and sea cucumbers; slight viewpoint shift.'),
        (7,43,'Same two divers with silver tanks and blue fins over matching reef; pair 43 has wider framing.')]:
        bridges.append(dict(left=key(links[a-1]['left']),right=key(links[b-1]['left']),
            review_status='visually_reviewed',verdict='same_scene_family_supported',finding=finding,
            evidence_pair_indices=[a,b],provenance='explicit_cross_sheet_visual_comparison'))
    prior=read(repo/'outputs/ruod-v8-refresh-inputs/reviewed_edges.json')
    cache={key(r):r for r in read(repo/'outputs/ruod-v8-refresh-inputs/fingerprints.json')}
    all_edges=prior+decisions+bridges
    groups=components([dict(left=cache[e['left']],right=cache[e['right']]) for e in all_edges])
    base={s:read(repo/'outputs/ruod-v8-refresh-base/annotations'/('instances_'+s+'.json')) for s in ('train','validation','test')}
    current={('test' if s=='test' else 'train')+':'+str(i['id']):s for s,d in base.items() for i in d['images']}
    actions=[]
    for n,g in enumerate(groups):
        live={k:current[k] for k in g['members'] if k in current}
        if 'test' in live.values():
            target='exclude'
        elif 'validation' in live.values():
            target='validation'
        else:
            target='train'
        for k,s in live.items():
            if s!='test' and s!=target:
                actions.append(dict(member=k,from_split=s,to_split=target,family_index=n))
    changes=Counter((a['from_split'],a['to_split']) for a in actions)
    counts={s:len(d['images']) for s,d in base.items()}
    for a in actions:
        counts[a['from_split']]-=1
        if a['to_split']!='exclude': counts[a['to_split']]+=1
    summary=dict(pairs_reviewed=130,sheets_inspected=33,local_source_hashes_checked=len(inventory['members']),
        supported_pairs=130,rejected_pairs=0,uncertain_pairs=0,explicit_cross_sheet_bridges=bridges,
        proposed_action_counts=[dict(from_split=a,to_split=b,count=n) for (a,b),n in sorted(changes.items())],
        proposed_images=counts,current_split='v8-reviewed',membership_changed=False,training_ready=False,
        limitation='Only inspected priority pairs and four explicit bridges are decided. Weak candidates, further same-scene bridges and conflict coverage remain open. Source hashes must be checked in Colab before application.')
    write(out/'visual_review.json',dict(summary,inventory_sha256=digest_file(out/'inventory.json'),
        prior_edges_sha256=digest_file(repo/'outputs/ruod-v8-refresh-inputs/reviewed_edges.json'),
        decisions=decisions,proposed_actions=actions))
    write(out/'review_summary.json',summary)
    print(summary)

if __name__=='__main__': main()
