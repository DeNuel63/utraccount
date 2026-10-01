"""Prepare all 130 v8 priority pairs for one consolidated visual review."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .apply_ruod_visual_families import read, write, digest_file

def main():
    repo=Path(__file__).resolve().parents[1]
    source=repo/'outputs/ruod-v8-workload/actionable_priority_pairs.json'
    edges=read(source)
    edges=sorted(edges,key=lambda e:(e['review_source']!='historical_family_connection',e['score'],e['left']['id'],e['right']['id']))
    out=repo/'outputs/ruod-v8-priority-review'; out.mkdir(exist_ok=True)
    members={r['origin']+':'+str(r['id']):r for e in edges for r in (e['left'],e['right'])}
    for r in members.values():
        if digest_file(Path(r['path']))!=r['source_sha256']:
            raise ValueError('Changed original: '+r['path'])
    font=ImageFont.truetype('arial.ttf',16)
    for start in range(0,len(edges),4):
        page=edges[start:start+4]
        canvas=Image.new('RGB',(1280,len(page)*330),'white'); draw=ImageDraw.Draw(canvas)
        for n,e in enumerate(page):
            y=n*330
            draw.text((10,y),'Pair {} | score {} | {}'.format(start+n+1,e['score'],e['review_source']),font=font,fill='black')
            for col,side in enumerate(('left','right')):
                r=e[side]; x=col*640
                draw.text((x+8,y+22),'{} | {}:{} | {}'.format(r['split'],r['origin'],r['id'],r['file_name']),font=font,fill='black')
                with Image.open(r['path']) as im:
                    im=im.convert('RGB'); im.thumbnail((620,280))
                    canvas.paste(im,(x+(640-im.width)//2,y+46+(280-im.height)//2))
        canvas.save(out/('pairs_{:02d}.png'.format(start//4+1)))
    write(out/'inventory.json',dict(base_split='v8-reviewed',source_sha256=digest_file(source),
        links=edges,members=members,hashes_checked=len(members),membership_changed=False))
    print('Pairs:',len(edges),'source hashes:',len(members),'pages:',(len(edges)+3)//4)

if __name__=='__main__':
    main()
