"""CPU exemplar texture repair; approved foreground and trajectory stay fixed."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from run_warp_control import translated

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'runs/approved_mask_cpu_20260929T221338371751Z'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def exemplar_fill(image,repair_mask,radius=5):
    """Match patches to visible original background; no fabricated geometry claim."""
    arr=np.asarray(image,dtype=np.float32).copy()/255
    hole=np.asarray(repair_mask)>0
    h,w=hole.shape;k=2*radius+1
    windows=np.lib.stride_tricks.sliding_window_view(arr,(k,k),axis=(0,1)).transpose(0,1,3,4,2)
    blocked=np.lib.stride_tricks.sliding_window_view(hole,(k,k)).any(axis=(-2,-1))
    cy,cx=np.mgrid[radius:h-radius:3,radius:w-radius:3]
    valid=~blocked[cy-radius,cx-radius];cy,cx=cy[valid],cx[valid]
    if len(cy)<100:raise ValueError('Insufficient clean donor background')
    donors=windows[cy-radius,cx-radius].copy()
    known=~hole;confidence=known.astype(np.float32)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    spatial=np.exp(-(xx*xx+yy*yy)/(radius*radius)).astype(np.float32)
    iterations=0;trace=[]
    while not known.all():
        neighborhood=np.zeros((h,w),dtype=np.int16)
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                neighborhood+=np.roll(known,(dy,dx),(0,1))
        eligible=(~known).copy();eligible[:radius]=False;eligible[-radius:]=False
        eligible[:,:radius]=False;eligible[:,-radius:]=False
        if not eligible.any():raise ValueError('Repair touches unsupported image border')
        score=np.where(eligible,neighborhood,-1)
        y,x=np.unravel_index(np.argmax(score),score.shape)
        sl=(slice(y-radius,y+radius+1),slice(x-radius,x+radius+1))
        weights=spatial*confidence[sl]*known[sl]
        if weights.sum()==0:raise ValueError('No matching context')
        target=arr[sl]
        distance=((donors-target)**2*weights[None,:,:,None]).sum(axis=(1,2,3))/(3*weights.sum())
        # Weak locality preference; matching appearance remains the main criterion.
        distance+=.002*((cy-y)**2+(cx-x)**2)/(h*h+w*w)
        best=int(np.argmin(distance));fill=~known[sl]
        arr[sl][fill]=donors[best][fill]
        confidence[sl][fill]=max(.1,float(confidence[sl].mean()))
        known[sl][fill]=True
        trace.append(dict(target=[int(x),int(y)],donor=[int(cx[best]),int(cy[best])]))
        iterations+=1
        if iterations>h*w:raise ValueError('Repair did not converge')
    result=np.rint(arr*255).clip(0,255).astype('uint8')
    original=np.asarray(image);result[~hole]=original[~hole]
    return Image.fromarray(result),trace

def main():
    before={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()}
    old=json.loads((BASE/'qc.json').read_text())
    image=Image.open(BASE/'keyframe.png').convert('RGB');mask=Image.open(BASE/'approved_mask.png').convert('L')
    assert hashlib.sha256(mask.tobytes()).hexdigest()==old['mask_pixels_sha256']
    moves=[tuple(f['translation_pixels']) for f in old['frames']]
    assert moves==[(0,0),(5,3),(9,6),(14,9),(19,11)]
    # Repair support is wider than foreground selection to remove residual edge colour.
    # This does NOT change the mask used to move the animal.
    repair=mask.filter(ImageFilter.MaxFilter(13))
    print('CPU exemplar repair: 11x11 donor patches, six-pixel repair margin.',flush=True)
    background,trace=exemplar_fill(image,repair)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'runs'/('background_exemplar_'+stamp);out.mkdir(parents=True,exist_ok=False)
    for name in ('frames','masks','annotations','diagnostics'):(out/name).mkdir()
    background.save(out/'diagnostics/background_fill.png');repair.save(out/'diagnostics/repair_support.png')
    mask.save(out/'approved_mask.png')
    sheet=Image.new('RGB',(1280,584),'white');draw=ImageDraw.Draw(sheet)
    report=dict(status='STATIC_BACKGROUND_COMPARISON_REVIEW_REQUIRED',parent_run=BASE.name,
        parent_qc_sha256=sha(BASE/'qc.json'),mask_pixels_sha256=old['mask_pixels_sha256'],
        foreground_mask_changed=False,motion_changed=False,model_loaded=False,gpu_used=False,
        training_started=False,dataset_accepted=False,
        method='Deterministic exemplar matching from visible original background',
        repair_margin_pixels=6,patch_size=11,donor_stride=3,patch_iterations=len(trace),
        limitations=['Invented background texture, not recovery of true hidden scene.',
            'Only background repair changed; repair support expanded by six pixels beyond approved foreground.',
            'Translucent fin mixtures and rigid motion remain unresolved.'],frames=[])
    source=np.asarray(image);inside=np.asarray(mask)>0;yy,xx=np.where(inside)
    for i,(dx,dy) in enumerate(moves):
        frame,moved=translated(image,background,mask,dx,dy)
        arr=np.asarray(frame);moving=np.asarray(moved)>0
        assert np.array_equal(arr[yy+dy,xx+dx],source[yy,xx])
        untouched=~((np.asarray(repair)>0)|moving)
        assert np.array_equal(arr[untouched],source[untouched])
        if i==0:assert np.array_equal(arr,source)
        frame.save(out/'frames'/('%06d.png'%i));moved.save(out/'masks'/('%06d.png'%i))
        annotation=json.loads((BASE/'annotations'/('%06d.json'%i)).read_text())
        (out/'annotations'/('%06d.json'%i)).write_text(json.dumps(annotation,indent=2))
        oldframe=Image.open(BASE/'frames'/('%06d.png'%i)).convert('RGB')
        sheet.paste(oldframe,(i*256,32));sheet.paste(frame,(i*256,324))
        draw.text((i*256+4,8),'Old fill | frame %d'%i,fill='black')
        draw.text((i*256+4,300),'Exemplar fill | frame %d'%i,fill='black')
        report['frames'].append(dict(frame_index=i,translation_pixels=[dx,dy],
            source_mask_pixels_preserved=True,outside_repair_and_motion_preserved=True,
            sha256=sha(out/'frames'/('%06d.png'%i))))
    sheet.save(out/'comparison.png')
    clean=Image.new('RGB',(512,288),'white');d=ImageDraw.Draw(clean)
    clean.paste(Image.open(BASE/'diagnostics/background_fill.png'),(0,32));clean.paste(background,(256,32))
    d.text((4,8),'Old static repair',fill='black');d.text((260,8),'Exemplar static repair',fill='black')
    clean.save(out/'background_comparison.png')
    assert before=={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()}
    report['parent_files_unchanged']=True
    (out/'qc.json').write_text(json.dumps(report,indent=2))
    (out/'diagnostics/donor_trace.json').write_text(json.dumps(trace))
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    print('Saved run:',out.name,flush=True)
    print('All five foreground preservation checks PASS; mask/motion/parent unchanged.',flush=True)

if __name__=='__main__':main()
