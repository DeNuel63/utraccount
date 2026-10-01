"""Render hash-verified pairs for every priority v7 historical connection."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .apply_ruod_visual_families import read, write, digest_file

def main():
    repo=Path(__file__).resolve().parents[1]
    source=repo/'outputs/ruod-v7-workload/historical_family_candidates.json'
    edges=[e for e in read(source) if e['score']<=4]
    output=repo/'outputs/ruod-historical14-review-v7'
    output.mkdir(exist_ok=True)
    members={r['origin']+':'+str(r['id']):r for e in edges for r in (e['left'],e['right'])}
    for r in members.values():
        if digest_file(Path(r['path']))!=r['source_sha256']:
            raise ValueError('Source changed: '+r['path'])
    font=ImageFont.truetype('arial.ttf',17)
    for start in range(0,len(edges),4):
        page=edges[start:start+4]
        canvas=Image.new('RGB',(1280,350*len(page)),'white')
        draw=ImageDraw.Draw(canvas)
        for index,e in enumerate(page):
            y=index*350
            draw.text((10,y+2),'Link {} | historical family {} | candidate, NOT verdict'.format(start+index+1,e['reviewed_family_index']),font=font,fill='black')
            for col,side in enumerate(('left','right')):
                r=e[side]; x=col*640
                draw.text((x+8,y+25),'{} | {}:{} | {}'.format(r['split'],r['origin'],r['id'],r['file_name']),font=font,fill='black')
                with Image.open(r['path']) as im:
                    im=im.convert('RGB'); im.thumbnail((620,294))
                    canvas.paste(im,(x+(640-im.width)//2,y+52+(294-im.height)//2))
        canvas.save(output/('pairs_{:02d}.png'.format(start//4+1)))
    write(output/'inventory.json',dict(base_split='v7-reviewed',candidate_source_sha256=digest_file(source),
        links=edges,members=members,local_original_hashes='PASS',hashes_checked=len(members),
        membership_changed=False))
    print('Links:',len(edges),'retained:',len({e['left']['id'] for e in edges}),'hashes:',len(members))
    for n,e in enumerate(edges,1):
        print(n,e['left']['split'],e['left']['id'],e['right']['id'],e['reviewed_family_index'])

if __name__=='__main__':
    main()
