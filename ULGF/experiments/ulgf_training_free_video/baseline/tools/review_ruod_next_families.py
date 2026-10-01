"""Prepare family-grouped candidate sheets after v2; presentation is not approval."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
try:
    from .apply_ruod_visual_families import read, write
except ImportError:
    from apply_ruod_visual_families import read, write


def prepare(screen, actions_path, output, prefix='B02', max_groups=6,
            boundaries=('train__test','validation__test'), max_score=None, test_linked_only=False):
    output.mkdir(parents=True,exist_ok=True)
    actions={a['member']:a['to_split'] for a in read(actions_path)}
    parent, nodes, edges = {}, {}, []
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key:
            parent[key]=find(parent[key])
        return parent[key]
    for boundary in boundaries:
        for row in read(screen/(boundary+'.json')):
            if max_score is not None and row['score'] > max_score:
                continue
            endpoint=[]
            for side in ('left','right'):
                member=dict(row[side])
                key=member['split']+':'+str(member['id'])
                canonical=('test' if member.get('origin',member['split'])=='test' else 'train')+':'+str(member['id'])
                current=actions.get(key,actions.get(canonical,member['split']))
                if current=='exclude':
                    break
                member['original_key']=key
                member['split']=current
                nodes[key]=member
                endpoint.append(key)
            if len(endpoint)!=2:
                continue
            a,b=endpoint
            if nodes[a]['split']==nodes[b]['split']:
                continue
            parent[find(a)]=find(b)
            edges.append(dict(left=a,right=b,phash_distance=row['phash_distance'],
                              dhash_distance=row['dhash_distance'],score=row['score']))
    groups={}
    for edge in edges:
        groups.setdefault(find(edge['left']),[]).append(edge)
    ordered=sorted(groups.values(),key=lambda g:(min(e['score'] for e in g),min(e['left'] for e in g)))
    if test_linked_only:
        ordered=[g for g in ordered if any(nodes[e[k]]['split']=='test' for e in g for k in ('left','right'))]
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:
        font=ImageFont.truetype('arial.ttf',17)
    manifest=[]
    for index,group in enumerate(ordered[:max_groups],1):
        keys={e[k] for e in group for k in ('left','right')}
        best=min(group,key=lambda e:(e['score'],e['left'],e['right']))
        chosen=[best['right'],best['left']]
        for edge in sorted(group,key=lambda e:(e['score'],e['left'],e['right'])):
            for key in (edge['left'],edge['right']):
                if key not in chosen:
                    chosen.append(key)
        name=prefix+'F{:02d}'.format(index)
        for start in range(0,len(chosen),8):
            page=chosen[start:start+8]
            sheet=Image.new('RGB',(1200,65+300*((len(page)+1)//2)),'white')
            draw=ImageDraw.Draw(sheet)
            draw.text((10,5),'{} | {} members, {} edges | page {}'.format(
                name,len(keys),len(group),start//8+1),font=font,fill='black')
            draw.text((10,30),'Grouping for review only. No automatic membership changes.',font=font,fill='black')
            for j,key in enumerate(page):
                member=nodes[key]
                x,y=(j%2)*600,65+(j//2)*300
                draw.text((x+8,y+2),'{} | {} | id {}'.format(member['split'],member['file_name'],member['id']),font=font,fill='black')
                with Image.open(member['path']) as image:
                    preview=image.convert('RGB')
                preview.thumbnail((580,270))
                sheet.paste(preview,(x+(600-preview.width)//2,y+25+(270-preview.height)//2))
            suffix='' if start==0 else '_page'+str(start//8+1)
            sheet.save(output/(name+suffix+'.png'))
        manifest.append(dict(candidate_family=name,total_members=len(keys),total_edges=len(group),
            shown_members={k:nodes[k] for k in chosen},
            shown_edges=[e for e in group if e['left'] in chosen and e['right'] in chosen],
            unshown_members=sorted(keys-set(chosen)),status='AWAITING_VISUAL_REVIEW'))
    write(output/'batch_candidates.json',manifest)
    print([(r['candidate_family'],r['total_members'],len(r['shown_members']),len(r['shown_edges'])) for r in manifest])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('screen','actions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    prepare(args.screen,args.actions,args.output)
