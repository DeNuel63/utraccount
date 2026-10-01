from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file

if __name__=='__main__':
    folder=Path(__file__).resolve().parents[1]/'outputs/ruod-historical22-review-v6'
    inventory=read(folder/'inventory.json')
    findings={1:'Matching scallops, sea cucumbers and foreground urchins on the same seabed.',
      2:'Same scallops, urchins and sea-cucumber layout.',3:'Same cuttlefish and coral panorama with small animal pose changes.',
      4:'Same seabed objects and scallop landmarks.',6:'Same seabed, scallops and foreground urchins.',
      7:'Same cuttlefish over matching coral panorama.',13:'Same cuttlefish/reef sequence, with changing pose.',
      14:'Same scallop/urchin/sea-cucumber arrangement.',17:'Same sea cucumbers and two urchins along diagonal seabed.',
      20:'Same scallop and foreground urchin scene.',36:'Same close cuttlefish, tentacles and coral background.'}
    retained={k:r for g in inventory for k,r in g['members'].items() if r['split'] in ('train','validation')}
    assert len(retained)==22 and all(r['split']=='train' for r in retained.values())
    unique={k:r for g in inventory for k,r in g['members'].items()}
    for r in unique.values():
        assert digest_file(Path(r['path']))==r['source_sha256']
    write(folder/'visual_review.json',dict(review_date='2026-09-25',base_split='v6-reviewed',
        inventory_sha256=digest_file(folder/'inventory.json'),membership_changed=False,
        reviewed_links=45,retained_images=22,local_hashes_checked=len(unique),
        decisions=[dict(family_index=g['family_index'],verdict='same_scene_family_supported',
            finding=findings[g['family_index']],reviewed_links=g['links']) for g in inventory],
        proposed_exclusions=sorted(retained),validation_changes=0,test_changes=0,
        limitation='Excluded anchors are evidence only; application needs v6-bound manifest and Colab hashes. No full leakage clearance.'))
    print('Review recorded: 22 train exclusions proposed; {} unique local hashes PASS'.format(len(unique)))
