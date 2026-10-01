"""Validate mask-derived annotations and package short/long CPU compositing controls."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, math, shutil, sys, zipfile
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from ulgf_video.schema import ClipManifest
from ulgf_video.validation import validate_manifest
PARENT=ROOT/'runs/reference_plate_cpu_20260929T223947096913Z'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def check_frame(index,frame,mask,annotation,source,base_mask,plate,offset):
    if frame.size!=mask.size or frame.size!=source.size or frame.size!=plate.size:raise ValueError('Dimensions differ')
    a=np.asarray(mask);base=np.asarray(base_mask)
    if not set(np.unique(a)).issubset({0,255}) or not (a>0).any():raise ValueError('Invalid binary mask')
    dx,dy=offset;ys,xs=np.where(base>0);h,w=a.shape
    if min(xs+dx)<0 or max(xs+dx)>=w or min(ys+dy)<0 or max(ys+dy)>=h:raise ValueError('Clipped mask')
    expected=np.zeros_like(a);expected[ys+dy,xs+dx]=255
    if not np.array_equal(a,expected):raise ValueError('Mask translation differs from trajectory')
    bbox=list(mask.getbbox())
    if annotation['frame_index']!=index or annotation['instance_id']!=1:raise ValueError('Frame or persistent ID mismatch')
    if (annotation['class_id'],annotation['class_name'])!=(4,'fish'):raise ValueError('Class mismatch')
    if annotation.get('bbox_max_exclusive') is not True:raise ValueError('Box convention mismatch')
    if annotation['bbox_xyxy_pixels']!=bbox:raise ValueError('Annotation does not bound translated mask')
    if not np.allclose(annotation['bbox_xyxy'],np.array(bbox)/[w,h,w,h],rtol=0,atol=1e-10):raise ValueError('Normalized box mismatch')
    if annotation['translation_pixels']!=list(offset):raise ValueError('Annotation motion mismatch')
    pixels=np.asarray(frame);fg=a>0
    if not np.array_equal(pixels[ys+dy,xs+dx],np.asarray(source)[ys,xs]):raise ValueError('Foreground altered')
    if not np.array_equal(pixels[~fg],np.asarray(plate)[~fg]):raise ValueError('Background altered')
    return dict(frame_index=index,instance_id=1,bbox_matches_mask=True,translation_matches=True,
                foreground_preserved=True,static_plate_preserved=True,mask_pixels=int(fg.sum()))

def long_offsets(count=48):
    if count<2:raise ValueError('Need at least two frames')
    return [(round(40*(.5-.5*math.cos(math.pi*i/(count-1)))),
             round(20*(.5-.5*math.cos(math.pi*i/(count-1))))) for i in range(count)]

def composite(source,base_mask,plate,offset):
    dx,dy=offset;box=base_mask.getbbox();x1,y1,x2,y2=box
    if not (0<=x1+dx<x2+dx<=plate.width and 0<=y1+dy<y2+dy<=plate.height):raise ValueError('Clipping rejected')
    frame=plate.copy();frame.paste(source.crop(box),(x1+dx,y1+dy),base_mask.crop(box))
    mask=Image.new('L',plate.size);mask.paste(base_mask.crop(box),(x1+dx,y1+dy))
    b=list(mask.getbbox())
    ann=dict(instance_id=1,class_id=4,class_name='fish',bbox_xyxy_pixels=b,
        bbox_xyxy=[v/256 for v in b],bbox_max_exclusive=True,translation_pixels=list(offset),
        provenance='Bounds of translated user-approved silhouette, not independent tracking ground truth')
    return frame,mask,ann

def playback(out,frames,fps):
    duration=round(1000/fps)
    frames[0].save(out/'playback.webp',save_all=True,append_images=frames[1:],
                   duration=[duration]*len(frames),loop=0,lossless=True,quality=100,method=4)
    # Repeated integer positions can be coalesced by the encoder. Validate by time, not frame count.
    total=0
    with Image.open(out/'playback.webp') as animation:
        for i in range(animation.n_frames):
            animation.seek(i);decoded=np.asarray(animation.convert('RGB'))
            ms=animation.info.get('duration')
            if ms is None or ms%duration:raise ValueError('Unexpected WebP timing')
            for tick in range(ms//duration):
                expected=total//duration
                if expected>=len(frames) or not np.array_equal(decoded,np.asarray(frames[expected])):
                    raise ValueError('WebP pixel/timing mismatch')
                total+=duration
    if total!=len(frames)*duration:raise ValueError('Playback duration mismatch')
    return dict(format='lossless animated WebP',fps=fps,duration_ms=total,
        interpolation=False,loop='Forward then reset; reset is not continuous movement',
        decoded_pixels_and_timing='PASS')

def build(out,source,mask,plate,moves,fps,existing=False):
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('frames','masks','annotations','provenance'):(out/folder).mkdir()
    checks=[];frames=[];manifest_frames=[]
    for i,offset in enumerate(moves):
        if existing:
            frame=Image.open(PARENT/'frames'/('%06d.png'%i)).convert('RGB')
            moved=Image.open(PARENT/'masks'/('%06d.png'%i)).convert('L')
            annotation=json.loads((PARENT/'annotations'/('%06d.json'%i)).read_text())
        else:
            frame,moved,annotation=composite(source,mask,plate,offset);annotation['frame_index']=i
        checks.append(check_frame(i,frame,moved,annotation,source,mask,plate,offset))
        if existing:
            for folder,ext in (('frames','png'),('masks','png'),('annotations','json')):
                shutil.copy2(PARENT/folder/('%06d.%s'%(i,ext)),out/folder/('%06d.%s'%(i,ext)))
        else:
            frame.save(out/'frames'/('%06d.png'%i));moved.save(out/'masks'/('%06d.png'%i))
            write_json(out/'annotations'/('%06d.json'%i),annotation)
        frames.append(frame)
        manifest_frames.append(dict(frame_index=i,image_path='frames/%06d.png'%i,
            objects=[dict(instance_id=1,class_id=4,class_name='fish',bbox_xyxy=annotation['bbox_xyxy'],visibility=1.0)]))
    spec=dict(schema_version='1.0',clip_id=out.name,seed=4,fps=fps,width=256,height=256,
        medium=dict(description='Static supplied plate; source-derived fish; rigid integer translation'),
        generation=dict(checkpoint='not_loaded_cpu_compositing',config='provenance/sequence.json',
            temporal_mode='source_pixel_rigid_translation',source_image='provenance/keyframe.png',
            motion=dict(offsets_pixels=[list(p) for p in moves])),frames=manifest_frames)
    manifest=ClipManifest.from_dict(spec);validate_manifest(manifest,out,require_images=True);manifest.write(out/'manifest.json')
    for filename in ('keyframe.png','approved_mask.png','clean_plate.png','qc.json'):
        shutil.copy2(PARENT/filename,out/'provenance'/filename)
    review=ROOT/'reviews/fish_mask_user_v3/mask_review.json'
    shutil.copy2(review,out/'provenance/mask_review_original.json')
    # Preserve the actual image-conditioning report alongside source-derived pixels.
    shutil.copy2(ROOT/'reviews/fish_crop_20260929T015815740178Z/report.json',out/'provenance/image_conditioning_report.json')
    info=dict(source_run=PARENT.name,source_run_qc_sha256=sha(PARENT/'qc.json'),
        annotation_semantics='Exact bounds of approved translated mask; not independently verified ground truth',
        mask_approval='User: The cyan boundary follows my intended outline',
        short_playback_approval='User: It is ok, lets move on',
        visual_approval='Accepted short diagnostic' if existing else 'PENDING longer trajectory review',
        model_loaded=False,gpu_used=False,training_started=False,dataset_accepted=False,
        limitations=['Source-derived compositing, not a learned video generator.',
          'Rigid motion and static background; no articulation, depth ordering or occlusion.',
          'Same planned identity and exact copied appearance, not a tracker evaluation.'],
        distinct_integer_offsets=len(set(moves)),frame_count=len(moves),fps=fps)
    write_json(out/'provenance/sequence.json',info)
    timing=playback(out,frames,fps)
    selected=sorted(set(round(i*(len(frames)-1)/5) for i in range(6)))
    sheet=Image.new('RGB',(len(selected)*256,288),'white');draw=ImageDraw.Draw(sheet)
    for col,i in enumerate(selected):
        sheet.paste(frames[i],(col*256,32));draw.text((col*256+4,8),'Frame %d | ID 1'%i,fill='black')
    sheet.save(out/'contact_sheet.png')
    write_json(out/'validation.json',dict(status='STRUCTURAL_AND_PIXEL_CHECKS_PASS',frame_checks=checks,
        persistent_id=1,canonical_manifest='PASS',playback=timing,visual_quality=info['visual_approval']))
    sums={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'SHA256SUMS.json',sums)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,out.name+'/'+str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        for name,expected in sums.items():
            if hashlib.sha256(z.read(out.name+'/'+name)).hexdigest()!=expected:raise ValueError('Archive verification failed')
    print(out.name+': '+str(len(frames))+' annotations/masks/IDs PASS; playback and ZIP hashes PASS',flush=True)

def main():
    before={str(p):sha(p) for p in PARENT.rglob('*') if p.is_file()}
    qc=json.loads((PARENT/'qc.json').read_text())
    for f in qc['frames']:
        assert sha(PARENT/'frames'/('%06d.png'%f['frame_index']))==f['frame_sha256']
    source=Image.open(PARENT/'keyframe.png').convert('RGB');mask=Image.open(PARENT/'approved_mask.png').convert('L')
    plate=Image.open(PARENT/'clean_plate.png').convert('RGB')
    assert sha(PARENT/'keyframe.png')==qc['keyframe_sha256']
    assert sha(PARENT/'clean_plate.png')==qc['clean_plate_sha256']
    assert hashlib.sha256(mask.tobytes()).hexdigest()==qc['mask_pixels_sha256']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=ROOT/'runs'/('sequence_delivery_'+stamp);root.mkdir(parents=True,exist_ok=False)
    build(root/'sequence_5frame',source,mask,plate,[tuple(f['translation_pixels']) for f in qc['frames']],4,True)
    build(root/'sequence_48frame',source,mask,plate,long_offsets(),10)
    assert all(sha(Path(p))==h for p,h in before.items())
    print('DELIVERY:',root.name,'Parent untouched.',flush=True)

if __name__=='__main__':main()
