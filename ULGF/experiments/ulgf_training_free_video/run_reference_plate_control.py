"""Deterministic CPU compositing with user-supplied clean plate; no synthesis."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,zipfile
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'runs/approved_mask_cpu_20260929T221338371751Z'
PREVIOUS=ROOT/'runs/background_exemplar_20260929T222245437781Z'
REFERENCE=Path('C:/Users/DE_NUE~1/AppData/Local/Temp/codex-clipboard-1c3b44dc-c62a-4ce6-b1c5-55939c335ce3.png')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    preserved={str(p):sha(p) for folder in (BASE,PREVIOUS) for p in folder.rglob('*') if p.is_file()}
    qc=json.loads((BASE/'qc.json').read_text())
    source=Image.open(BASE/'keyframe.png').convert('RGB')
    mask=Image.open(BASE/'approved_mask.png').convert('L')
    assert hashlib.sha256(mask.tobytes()).hexdigest()==qc['mask_pixels_sha256']
    assert source.size==mask.size==(256,256)
    with Image.open(REFERENCE) as ref:
        if ref.getexif().get(274,1) not in (None,1):raise ValueError('Unexpected plate orientation')
        w,h=ref.size;side=min(w,h);left=(w-side)//2;top=(h-side)//2
        crop=(left,top,left+side,top+side)
        plate=ref.convert('RGB').crop(crop).resize((256,256),Image.Resampling.LANCZOS)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'runs'/('reference_plate_cpu_'+stamp);out.mkdir(parents=True,exist_ok=False)
    for name in ('frames','masks','annotations'):(out/name).mkdir()
    # Preserve supplied plate bytes before temporary attachment storage disappears.
    (out/'supplied_plate.png').write_bytes(REFERENCE.read_bytes())
    plate.save(out/'clean_plate.png');mask.save(out/'approved_mask.png');source.save(out/'keyframe.png')
    sheet=Image.new('RGB',(1280,292),'white');comparison=Image.new('RGB',(1280,584),'white')
    annotated=sheet.copy();draw=ImageDraw.Draw(sheet);compdraw=ImageDraw.Draw(comparison)
    report=dict(status='STATIC_PLATE_CONTROL_COMPLETE_VISUAL_REVIEW_REQUIRED',
        training_started=False,model_loaded=False,gpu_used=False,dataset_accepted=False,
        plate_origin='User-supplied reference used as a replacement background, not reconstructed hidden geometry',
        supplied_plate_sha256=sha(REFERENCE),supplied_plate_size=[w,h],crop_xyxy=list(crop),
        clean_plate_sha256=sha(out/'clean_plate.png'),keyframe_sha256=sha(BASE/'keyframe.png'),
        mask_pixels_sha256=qc['mask_pixels_sha256'],foreground_mask_changed=False,motion_changed=False,
        plate_visual_review='Assistant inspected: no obvious fish, propagation streaks or patch seams; user review pending',
        limitations=['Rigid 2D translation, not swimming or learned temporal generation.',
          'Translucent fin pixels retain original background mixtures.',
          'No depth-aware occlusion; approved silhouette bounds are not independently verified ground truth.',
          'Frame 0 intentionally recomposited on the same replacement plate as later frames.'],frames=[])
    original=np.asarray(source);platearr=np.asarray(plate);yy,xx=np.where(np.asarray(mask)>0)
    bbox=mask.getbbox();x1,y1,x2,y2=bbox
    for i,item in enumerate(qc['frames']):
        dx,dy=item['translation_pixels']
        assert 0<=x1+dx<x2+dx<=256 and 0<=y1+dy<y2+dy<=256
        frame=plate.copy();frame.paste(source.crop(bbox),(x1+dx,y1+dy),mask.crop(bbox))
        moved=Image.new('L',(256,256));moved.paste(mask.crop(bbox),(x1+dx,y1+dy))
        fg=np.asarray(moved)>0;pixels=np.asarray(frame)
        assert np.array_equal(pixels[yy+dy,xx+dx],original[yy,xx])
        assert np.array_equal(pixels[~fg],platearr[~fg])
        frame.save(out/'frames'/('%06d.png'%i));moved.save(out/'masks'/('%06d.png'%i))
        ann=json.loads((BASE/'annotations'/('%06d.json'%i)).read_text())
        (out/'annotations'/('%06d.json'%i)).write_text(json.dumps(ann,indent=2))
        sheet.paste(frame,(i*256,36));draw.text((i*256+4,10),'Frame %d | dx=%d dy=%d'%(i,dx,dy),fill='black')
        comparison.paste(Image.open(PREVIOUS/'frames'/('%06d.png'%i)),(i*256,36))
        comparison.paste(frame,(i*256,328))
        compdraw.text((i*256+4,10),'Previous patch repair | %d'%i,fill='black')
        compdraw.text((i*256+4,302),'Supplied static plate | %d'%i,fill='black')
        overlay=frame.copy();d=ImageDraw.Draw(overlay);b=moved.getbbox()
        d.rectangle((b[0],b[1],b[2]-1,b[3]-1),outline='red')
        d.text((b[0],max(0,b[1]-12)),'fish ID 1',fill='red')
        annotated.paste(overlay,(i*256,36))
        report['frames'].append(dict(frame_index=i,translation_pixels=[dx,dy],
            foreground_pixels_preserved=True,all_unmasked_pixels_equal_static_plate=True,
            frame_sha256=sha(out/'frames'/('%06d.png'%i))))
    sheet.save(out/'contact_sheet.png');comparison.save(out/'comparison.png');annotated.save(out/'annotated_contact_sheet.png')
    assert all(sha(Path(p))==h for p,h in preserved.items())
    report['previous_runs_unchanged']=True
    (out/'qc.json').write_text(json.dumps(report,indent=2))
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    print('Saved:',out.name)
    print('PASS: five exact foreground checks; five exact static-plate checks; previous runs unchanged.')

if __name__=='__main__':main()
