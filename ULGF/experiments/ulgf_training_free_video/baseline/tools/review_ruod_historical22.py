"""Present every strong historical connection for the 22 retained images."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .apply_ruod_visual_families import read, write, digest_file

if __name__=='__main__':
    repo=Path(__file__).resolve().parents[1]
    folder=repo/'outputs/ruod-historical22-review-v6'
    folder.mkdir(exist_ok=True)
    edges=[e for e in read(repo/'outputs/ruod-v6-workload/historical_family_candidates.json') if e['score']<=4]
    families={}
    for e in edges:
        families.setdefault(e['reviewed_family_index'],[]).append(e)
    font=ImageFont.truetype('arial.ttf',17)
    inventory=[]
    for family,links in sorted(families.items()):
        rows={r['origin']+':'+str(r['id']):r for e in links for r in (e['left'],e['right'])}
        for r in rows.values():
            if digest_file(Path(r['path']))!=r['source_sha256']:
                raise ValueError('Source changed')
        ordered=sorted(rows.values(),key=lambda r:(r['split'] not in ('test','exclude'),r['split'],r['id']))
        for start in range(0,len(ordered),8):
            page=ordered[start:start+8]
            canvas=Image.new('RGB',(1200,60+300*((len(page)+1)//2)),'white')
            draw=ImageDraw.Draw(canvas)
            draw.text((10,8),'Historical family {} | page {} | excluded images are reference ONLY'.format(family,start//8+1),font=font,fill='black')
            for j,r in enumerate(page):
                x,y=(j%2)*600,60+(j//2)*300
                draw.text((x+8,y),'{} | {} | {}:{}'.format(r['split'],r['file_name'],r['origin'],r['id']),font=font,fill='black')
                with Image.open(r['path']) as im:
                    im=im.convert('RGB'); im.thumbnail((580,270))
                    canvas.paste(im,(x+(600-im.width)//2,y+25))
            canvas.save(folder/('family_{}_page{}.png'.format(family,start//8+1)))
        inventory.append(dict(family_index=family,members=rows,links=links))
    write(folder/'inventory.json',inventory)
    print([(r['family_index'],len(r['members']),len(r['links'])) for r in inventory])
