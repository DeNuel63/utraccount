"""Record inspected batch-05 scenes, including reviewed prior-family bridge."""
from collections import Counter
from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file

FINDINGS = [
 'Same seabed layout, scallops, sea cucumbers and foreground urchins across all four views.',
 'Matching aquarium corals and fish composition.',
 'Same cuttlefish pose, body markings and surrounding rocks.',
 'Same turtle reaching toward jellyfish, matching flippers and background.',
 'Same two lionfish against matching rock and coral; small fin changes indicate a related sequence.',
 'Same turtle shell markings and reef ledge, purple branches and background fish across the sequence. Visually confirmed bridge from test:2263 to validation:7182 connects the prior B04F12 family; all 14 prior members are now validation in v5.',
 'Same cuttlefish above matching panoramic coral reef; small position changes support same-scene grouping.',
 'Same resting turtle, flipper markings and surrounding coral bowl; background fish move between related views.',
 'Same close-up turtle head markings, eye, neck and rock background.',
 'Same aquarium coral arrangement and fish positions.',
 'Same aquarium coral arrangement with slightly moving fish and framing differences.',
 'Same turtle and jellyfish, matching front flippers and rocky backdrop.'
]

def main():
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-near-review-batch05-v5'
    original=folder/'priority_candidates.json'
    if not original.exists():
        write(original,read(folder/'batch_candidates.json'))
    candidates=read(original)
    prior=read(repo/'outputs/ruod-near-review-batch04-v4/batch_candidates.json')[-1]
    assert prior['candidate_family']=='B04F12'
    actions={a['member']:a['to_split'] for a in read(folder/'v5_cumulative_actions.json')}
    target=candidates[5]
    assert target['candidate_family']=='B05F06'
    mapping={}
    for old,row in prior['shown_members'].items():
        canonical='train:'+str(row['id'])
        current=actions.get(canonical,row['split'])
        assert current=='validation'
        new='validation:'+str(row['id'])
        mapping[old]=new
        target['shown_members'][new]=dict(row,split=current,original_key=old)
    for edge in prior['shown_edges']:
        target['shown_edges'].append(dict(edge,left=mapping[edge['left']],right=mapping[edge['right']],
            provenance='B04F12 visually reviewed edges, re-inspected with current test bridge'))
    target['shown_edges'].append(dict(left='test:2263',right='validation:7182',
        provenance='Explicit visual bridge; not inferred from hash threshold',
        finding='Matching turtle shell/pose, reef ledge and branching purple coral; framing/border differs.'))
    target['total_members']=len(target['shown_members'])
    target['total_edges']=len(target['shown_edges'])
    target['supplemental_evidence']='B04F12.png and B04F12_page2.png re-inspected; old sheet labels predate v5 moves.'
    write(folder/'batch_candidates.json',candidates)
    reviews,proposals,members=[],[],{}
    for group,finding in zip(candidates,FINDINGS):
        for k,row in group['shown_members'].items():
            canonical=row['origin']+':'+str(row['id'])
            assert actions.get(canonical)!='exclude'
            assert digest_file(Path(row['path']))==row['source_sha256']
            members[canonical]=row
            if row['split']!='test':
                proposals.append(dict(member=row['split']+':'+str(row['id']),from_split=row['split'],
                    to_split='exclude',family=group['candidate_family']))
        reviews.append(dict(candidate_family=group['candidate_family'],members_reviewed=group['total_members'],
            links_reviewed=group['total_edges'],verdict='same_scene_family_supported',finding=finding))
    counts=Counter(p['from_split'] for p in proposals)
    report=dict(review_date='2026-09-24',current_split='v5-reviewed',membership_changed=False,
        review_status='BATCH_VISUAL_REVIEW_COMPLETE',batch_candidates_sha256=digest_file(folder/'batch_candidates.json'),
        priority_candidates_sha256=digest_file(original),reviews=reviews,reviewed_members=len(members),
        reviewed_links=sum(r['links_reviewed'] for r in reviews),
        local_original_file_hash_checks='PASS',proposed_actions=proposals,exclusions_by_split=dict(counts),
        prior_family_reassessment='B04F12 must now be excluded from validation because explicit reviewed bridge connects it to test; no prior excluded image is reintroduced.',
        limitations='Only selected priority groups and one explicit prior-family bridge reviewed. Additional weaker links and other same-scene members may remain. No split application or full-clearance claim.')
    write(folder/'visual_review.json',report)
    print(dict(members=len(members),links=report['reviewed_links'],exclusions=dict(counts)))

if __name__=='__main__':
    main()
