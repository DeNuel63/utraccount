"""Record explicit visual decisions after inspecting all 14 batch-04 sheets."""
from collections import Counter
from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file

FINDINGS = [
    ('near_identical_photo_supported', 'Same turtle and jellyfish, matching shell markings, flippers and framing.'),
    ('same_scene_family_supported', 'Same swimming turtle and shallow reef; all three views share pose and coral landmarks with minor framing differences.'),
    ('near_identical_photo_supported', 'Same branching coral, purple sponge, yellow fish and sand landmarks.'),
    ('same_scene_family_supported', 'Same sea cucumber/urchin arrangement and seabed across all three views.'),
    ('near_identical_photo_supported', 'Same close diver/turtle poses and distant diver at upper left.'),
    ('near_identical_photo_supported', 'Same frontal turtle, flipper positions and reef beneath the surface.'),
    ('near_identical_photo_supported', 'Same diver, surrounding fish and colorful coral composition.'),
    ('near_identical_photo_supported', 'Same clustered sea cucumbers, urchins and seabed marks.'),
    ('near_identical_photo_supported', 'Same aquarium coral structure and fish positions.'),
    ('near_identical_photo_supported', 'Same schooling fish and branching foreground coral.'),
    ('same_scene_family_supported', 'All sixteen views share foreground urchins, bright scallops and surrounding sea cucumbers. Minor pose/framing changes support a scene family, not exact pixel identity.'),
    ('same_scene_family_supported', 'All fourteen views show the same turtle over the same reef and branching coral. Moving background fish and small pose changes indicate a related sequence. Keep together, not across train/validation.')]

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    folder = repo / 'outputs/ruod-near-review-batch04-v4'
    candidates = read(folder / 'batch_candidates.json')
    prior = {a['member']: a['to_split'] for a in read(folder / 'v4_cumulative_actions.json')}
    assert len(candidates) == len(FINDINGS)
    reviews, actions, unique = [], [], set()
    for group, (verdict, finding) in zip(candidates, FINDINGS):
        members = group['shown_members']
        assert not group['unshown_members']
        assert len(members) == group['total_members']
        splits = {m['split'] for m in members.values()}
        target = 'exclude' if 'test' in splits else 'validation'
        for key, member in members.items():
            canonical = ('test' if member['origin'] == 'test' else 'train') + ':' + str(member['id'])
            assert prior.get(canonical) != 'exclude'
            assert digest_file(Path(member['path'])) == member['source_sha256']
            unique.add(canonical)
            if member['split'] != 'test' and member['split'] != target:
                actions.append(dict(member=member['split']+':'+str(member['id']),
                                    from_split=member['split'], to_split=target,
                                    family=group['candidate_family']))
        reviews.append(dict(candidate_family=group['candidate_family'],
                            members_reviewed=len(members), links_reviewed=len(group['shown_edges']),
                            verdict=verdict, finding=finding))
    counts = Counter(a['from_split']+'->'+a['to_split'] for a in actions)
    report = dict(review_date='2026-09-24', current_split='v4-reviewed',
        batch_candidates_sha256=digest_file(folder / 'batch_candidates.json'),
        review_status='BATCH_VISUAL_REVIEW_COMPLETE', membership_changed=False,
        reviews=reviews, reviewed_members=len(unique), reviewed_links=sum(r['links_reviewed'] for r in reviews),
        local_original_file_hash_checks='PASS', resolved_excluded_members_included=0,
        proposed_action_counts=dict(counts), proposed_actions=actions,
        limitations='Only 12 strongest groups from saved top-200-per-boundary shortlists, not all 6274 candidates. No exhaustive near-duplicate clearance. Colab hashes and v4 contracts must be verified before changes.')
    write(folder / 'visual_review.json', report)
    print(dict(members=len(unique), links=report['reviewed_links'], actions=dict(counts)))
