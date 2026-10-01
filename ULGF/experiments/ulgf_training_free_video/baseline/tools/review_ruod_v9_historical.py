"""Render retained candidates and complete shortest reviewed paths to test."""
from pathlib import Path
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from .apply_ruod_visual_families import read, write, digest_file

def main():
    repo=Path(__file__).resolve().parents[1]
    inputs=repo/'outputs/ruod-v9-refresh-inputs'
    pairs_path=repo/'outputs/ruod-v9-workload/actionable_priority_pairs.json'
    edges=read(inputs/'reviewed_edges.json'); graph={}
    for e in edges:
        a,b=e['left'],e['right']; graph.setdefault(a,set()).add(b); graph.setdefault(b,set()).add(a)
    rows={r['origin']+':'+str(r['id']):r for r in read(inputs/'fingerprints.json')}
    out=repo/'outputs/ruod-v9-historical-review'; out.mkdir(exist_ok=True)
    records=[]; members={}; font=ImageFont.truetype('arial.ttf',17)
    for n,e in enumerate(read(pairs_path),1):
        left=e['left']['origin']+':'+str(e['left']['id']); anchor=e['right']['origin']+':'+str(e['right']['id'])
        pending=deque([[anchor]]); seen={anchor}; path=None
        while pending:
            trial=pending.popleft()
            if trial[-1].startswith('test:'): path=trial; break
            for k in sorted(graph.get(trial[-1],set())-seen):
                seen.add(k); pending.append(trial+[k])
        if path is None: raise ValueError('No reviewed path to test')
        nodes=[left]+path
        canvas=Image.new('RGB',(1500,((len(nodes)+2)//3)*370),'white'); draw=ImageDraw.Draw(canvas)
        for i,k in enumerate(nodes):
            r=rows[k]; p=repo.parent/'RUOD/RUOD_pic'/r['origin']/r['file_name']
            if digest_file(p)!=r['source_sha256']: raise ValueError('Source changed: '+str(p))
            members[k]=dict(r,path=str(p))
            x=(i%3)*500; y=(i//3)*370
            label='RETAINED' if i==0 else ('OFFICIAL TEST' if k.startswith('test:') else 'HISTORICAL ANCHOR')
            draw.text((x+8,y+8),'Pair {} | {} | {}'.format(n,label,k),font=font,fill='black')
            draw.text((x+8,y+32),r['file_name'],font=font,fill='black')
            with Image.open(p) as im:
                im=im.convert('RGB'); im.thumbnail((480,310)); canvas.paste(im,(x+(500-im.width)//2,y+55+(310-im.height)//2))
        filename='context_{:02d}.png'.format(n); canvas.save(out/filename)
        records.append(dict(pair_index=n,left=left,right=anchor,anchor_to_test_reviewed_path=path,sheet=filename))
    write(out/'inventory.json',dict(base_split='v9-reviewed',pairs_sha256=digest_file(pairs_path),
        prior_edges_sha256=digest_file(inputs/'reviewed_edges.json'),records=records,members=members,
        hashes_checked=len(members),membership_changed=False))
    print('Pages:',len(records),'unique evidence hashes:',len(members))

if __name__=='__main__': main()
