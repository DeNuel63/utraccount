"""Record manual review of eleven rendered candidate-to-test context sheets."""
from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file

def main():
    repo=Path(__file__).resolve().parents[1]; folder=repo/'outputs/ruod-v9-historical-review'
    inventory=read(folder/'inventory.json')
    findings={
        1:'Same central pale shells, foreground angular fragment, urchins and sea-cucumber arrangement across candidate, anchor and test; small blur/pose differences.',
        2:'Same open sandy left area, two right-edge urchins and foreground sea-cucumber cluster across all three images.',
        3:'Same foreground urchins, curled sea cucumber, shells and background arrangement along the complete four-image path.',
        4:'Same bright central shells, foreground angular fragment and surrounding urchin/sea-cucumber positions along the five-image path.',
        5:'Same foreground urchins and surrounding sea-cucumber/shell arrangement along the four-image path; modest framing differences.',
        6:'Same lower/right urchins, pink shell, central sea cucumbers and background seabed; slight viewpoint changes.',
        7:'Same large lower-right urchin, pink central shell and distinctive sea-cucumber arrangement across candidate, anchor and test.',
        8:'Same diver with silver tank and camera, turtle and distinctive plate-coral/rock arrangement; small animal/pose changes.',
        9:'Same candidate as pair 4; second anchor and remaining path preserve the central pale shells, urchins and foreground angular fragment.',
        10:'Same scene as pair 1 with matching central pale shells, foreground angular fragment and animals; modest framing/blur differences.',
        11:'Same green-water seabed, lower foreground mesh/equipment and diagonal brown bar; seabed landmarks agree despite transient particles.'}
    for r in inventory['members'].values():
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Evidence changed')
    decisions=[dict(r,review_status='visually_reviewed',verdict='same_scene_family_supported',
                    finding=findings[r['pair_index']]) for r in inventory['records']]
    excluded=sorted({r['left'] for r in decisions})
    assert len(decisions)==11 and len(excluded)==10
    # Explicit cross-sheet scene connections observed during this review; these
    # connect already test-linked families, adding no further proposed removals.
    by_index={r['pair_index']:r for r in decisions}
    bridges=[]
    for a,b,finding in [(1,4,'Same bright shell cluster and angular foreground fragment with matching surrounding animals.'),
                        (3,6,'Same foreground urchins, central shells and sea-cucumber layout under changed framing.'),
                        (6,7,'Same pink shell and surrounding urchin/sea-cucumber layout under changed framing.')]:
        bridges.append(dict(left=by_index[a]['left'],right=by_index[b]['left'],finding=finding,
            evidence_pair_indices=[a,b],review_status='visually_reviewed',verdict='same_scene_family_supported'))
    write(folder/'visual_review.json',dict(base_split='v9-reviewed',inventory_sha256=digest_file(folder/'inventory.json'),
        prior_edges_sha256=inventory['prior_edges_sha256'],decisions=decisions,explicit_cross_sheet_bridges=bridges,
        proposed_exclusions=excluded,proposed_moves=[],proposed_images=dict(train=7540,validation=847,test=4200),
        original_hashes_checked=30,membership_changed=False,training_ready=False,
        limitation='Only these ten retained images reviewed. Colab hash verification/application, weaker-candidate review and exclusion coverage remain pending.'))
    print('Recorded 11 supported links, ten proposed training exclusions, zero moves. No split changes.')

if __name__=='__main__': main()
