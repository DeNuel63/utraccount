"""Record the completed 16-pair visual inspection; do not apply exclusions."""
from pathlib import Path
from .apply_ruod_visual_families import read, write, digest_file

def main():
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-historical14-review-v7'
    inventory=read(folder/'inventory.json')
    findings={
        20:'Same cuttlefish above matching branching coral panorama; small pose/framing difference.',
        6:'Matching foreground urchin, central scallop shells and surrounding sea-cucumber arrangement; slight viewpoint shifts.',
        32:'Matching central shells, upper urchin and foreground sea cucumbers; slight camera/framing differences across the sequence.',
        12:'Matching cuttlefish, left branching coral and lower foreground coral; small animal pose change.',
        31:'Matching cuttlefish and coral landmarks across the entire background; small pose change.',
        11:'Matching bright upper shell, central flat shell, upper-right urchin and sea-cucumber arrangement.',
        9:'Matching bright central shell, surrounding urchins, lower flat shell and sea-cucumber positions.'}
    # Confirm prior recorded ancestry; excluded anchors remain evidence only.
    graph={}
    for e in read(repo/'outputs/ruod-v7-refresh-inputs/reviewed_edges.json'):
        for a,b in ((e['left'],e['right']),(e['right'],e['left'])):
            graph.setdefault(a,set()).add(b)
    def path_to_test(start):
        pending=[start]; paths={start:[start]}
        for node in pending:
            if node.startswith('test:'):
                return paths[node]
            for nxt in sorted(graph.get(node,set())):
                if nxt not in paths:
                    paths[nxt]=paths[node]+[nxt]; pending.append(nxt)
        raise ValueError('No previously reviewed path to official test: '+start)
    for r in inventory['members'].values():
        if digest_file(Path(r['path']))!=r['source_sha256']:
            raise ValueError('Source changed')
    decisions=[]
    for index,e in enumerate(inventory['links'],1):
        left=e['left']['origin']+':'+str(e['left']['id'])
        right=e['right']['origin']+':'+str(e['right']['id'])
        decisions.append(dict(link_index=index,left=left,right=right,
            review_status='visually_reviewed',verdict='same_scene_family_supported',
            finding=findings[e['reviewed_family_index']],
            anchor_to_test_reviewed_path=path_to_test(right),
            sheet='pairs_{:02d}.png'.format((index-1)//4+1)))
    proposed=sorted({d['left'] for d in decisions})
    if len(proposed)!=14 or len(decisions)!=16 or any(e['left']['split']!='train' for e in inventory['links']):
        raise ValueError('Unexpected review scope')
    write(folder/'visual_review.json',dict(base_split='v7-reviewed',review_date='2026-09-25',
        inventory_sha256=digest_file(folder/'inventory.json'),
        prior_reviewed_edges_sha256=digest_file(repo/'outputs/ruod-v7-refresh-inputs/reviewed_edges.json'),
        decisions=decisions,proposed_exclusions=proposed,local_source_hashes_checked=len(inventory['members']),
        sheets_inspected=4,validation_changes=0,test_changes=0,membership_changed=False,
        proposed_counts=dict(train=7656,validation=852,test=4200),
        limitation='Same-scene family findings, not byte-identity or proven capture chronology. Colab source hashes and v7 contracts must be verified before application. Full candidate clearance pending.'))
    print('16 reviewed links, 14 train exclusions proposed, 25 local hashes PASS; all anchors have recorded test ancestry.')

if __name__=='__main__':
    main()
