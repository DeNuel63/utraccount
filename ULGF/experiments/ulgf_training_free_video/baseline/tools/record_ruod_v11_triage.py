"""Persist manual diagnostic review; never apply membership changes."""
from pathlib import Path
from .apply_ruod_visual_families import read,write,digest_file

def main():
    repo=Path(__file__).resolve().parents[1]; out=repo/'outputs/ruod-v11-weak-triage'
    inv=read(out/'inventory.json')
    findings=[
        'Same two divers, silver tanks, foreground diver equipment/pose and reef background.',
        'Same foreground equipment, bright shell, right flat rock and branching seabed growth.',
        'Same diver and large purple jellyfish/tentacle geometry; crop/watermark difference.',
        'Same paired cuttlefish with distinctive tentacle poses and rock contours.',
        'Same two lionfish, fin rays and textured rock backdrop.',
        'Same three squid and distinctive white branching coral; animal poses shift between views.',
        'Same turtle, yellow fish positions and patterned coral reef; appearance/colour differences.',
        'Similar foreground equipment and green-water seabed, but low detail prevents a confident same-scene decision.',
        'Same foreground equipment and distinctive upright branching growth and surrounding rock contours.',
        'Same aquarium coral structures and yellow foreground fish composition.',
        'Same cuttlefish tentacle pose and diver goggles, regulator and equipment.',
        'Same three spotted fish, red left rock wall and blue central background opening.']
    decisions=[]
    for i,(e,f) in enumerate(zip(inv['pairs'],findings),1):
        key=lambda r:r['origin']+':'+str(r['id'])
        decisions.append(dict(left=key(e['left']),right=key(e['right']),pair_index=i,
            review_status='visually_reviewed',verdict='uncertain' if i==8 else 'same_scene_family_supported',
            finding=f,sheet='pairs_%02d.png'%((i-1)//4+1)))
    for r in inv['members'].values():
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Source changed')
    write(out/'visual_review.json',dict(base_split='v11-reviewed',inventory_sha256=digest_file(out/'inventory.json'),
        decisions=decisions,supported=11,uncertain=1,rejected=0,hashes_checked=24,membership_changed=False,
        training_ready=False,limitation='Diagnostic strongest-flag sample, not representative or full clearance. Expand reviewed family connections before proposing any actions.'))

if __name__=='__main__': main()
