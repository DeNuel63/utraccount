"""Render all weaker retained neighbors of the two priority historical families."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .apply_ruod_visual_families import read,write,digest_file
from .screen_ruod_near_duplicates import contact_sheet

def main():
    repo=Path(__file__).resolve().parents[1]; root=repo/'outputs/ruod-v10-workload'
    inputs=repo/'outputs/ruod-v10-refresh-inputs'; out=repo/'outputs/ruod-v10-expanded-review'; out.mkdir(exist_ok=True)
    families=read(root/'historical_reviewed_families.json')
    selected={e['reviewed_family_index'] for e in read(root/'actionable_priority_pairs.json')}
    candidates=[e for e in read(root/'historical_family_candidates.json') if e['reviewed_family_index'] in selected]
    key=lambda r:r['origin']+':'+str(r['id'])
    best={}
    for e in sorted(candidates,key=lambda e:e['score']): best.setdefault(key(e['left']),e)
    pairs=list(best.values())
    lookup={key(r):r for r in read(inputs/'fingerprints.json')}
    keys=set(best)|{k for i in selected for k in families[i]['members']}
    for k in keys:
        r=lookup[k]; r['path']=str(repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name'])
        if digest_file(Path(r['path']))!=r['source_sha256']: raise ValueError('Source changed')
    for i in range(0,len(pairs),4): contact_sheet(pairs,out/('pairs_%02d.png'%(i//4+1)),i,4)
    font=ImageFont.truetype('arial.ttf',17)
    for family in sorted(selected):
        nodes=families[family]['members']
        for start in range(0,len(nodes),8):
            part=nodes[start:start+8]; canvas=Image.new('RGB',(1600,((len(part)+3)//4)*280),'white'); draw=ImageDraw.Draw(canvas)
            for j,k in enumerate(part):
                r=lookup[k]; x=j%4*400; y=j//4*280
                draw.text((x+5,y+4),k+' | '+r['file_name'],font=font,fill='black')
                with Image.open(r['path']) as im:
                    im=im.convert('RGB'); im.thumbnail((390,245)); canvas.paste(im,(x+(400-im.width)//2,y+28))
            canvas.save(out/('family_%d_%02d.png'%(family,start//8+1)))
    write(out/'inventory.json',dict(base_split='v10-reviewed',selected_families=sorted(selected),
        pairs=pairs,all_candidate_edges=candidates,members={k:lookup[k] for k in sorted(keys)},
        prior_edges_sha256=digest_file(inputs/'reviewed_edges.json'),hashes_checked=len(keys),membership_changed=False))
    print('Retained candidates:',len(pairs),'evidence hashes:',len(keys),'sheets:',len(list(out.glob('*.png'))))

if __name__=='__main__': main()
