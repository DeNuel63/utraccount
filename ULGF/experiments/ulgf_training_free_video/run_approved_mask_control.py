"""Local CPU replay on approved user-guided mask; no model or training imports."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from run_warp_control import fill_hole, translated

ROOT=Path(__file__).resolve().parent
MOVES=[(0,0),(5,3),(9,6),(14,9),(19,11)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    review=ROOT/'reviews/fish_mask_user_v3'
    source=ROOT/'reviews/fish_crop_20260929T015815740178Z/strength_0.05.png'
    maskpath=review/'mask_draft.png'
    spec=json.loads((review/'mask_review.json').read_text())
    assert sha(source)==spec['keyframe_sha256']
    assert sha(maskpath)==spec['mask_file_sha256']
    original=Image.open(source).convert('RGB');mask=Image.open(maskpath).convert('L')
    assert original.size==mask.size==(256,256)
    assert set(np.unique(np.asarray(mask)))=={0,255}
    assert hashlib.sha256(mask.tobytes()).hexdigest()==spec['mask_pixels_sha256']
    snapshot=json.loads((ROOT/'baseline_snapshot.json').read_text())
    before={p:sha(ROOT/'baseline'/p) for p in snapshot['files']}
    assert before==snapshot['files']
    background=fill_hole(original,mask)
    rendered=[translated(original,background,mask,*move) for move in MOVES]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'runs'/('approved_mask_cpu_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('frames','masks','annotations','diagnostics'):(out/folder).mkdir()
    original.save(out/'keyframe.png');mask.save(out/'approved_mask.png')
    background.save(out/'diagnostics/background_fill.png')
    approval=dict(scope='Mask boundary matches user intended outline for diagnostic use only',
        user_statement='The cyan boundary follows my intended outline',
        recorded_utc=stamp,keyframe_sha256=sha(source),mask_pixels_sha256=spec['mask_pixels_sha256'],
        verified_ground_truth=False)
    (out/'mask_approval.json').write_text(json.dumps(approval,indent=2))
    report=dict(status='CPU_CONTROL_COMPLETE_ARTIFACT_REVIEW_REQUIRED',model_loaded=False,
        gpu_used=False,training_started=False,dataset_accepted=False,
        source_keyframe_sha256=sha(source),mask_file_sha256=sha(maskpath),
        mask_pixels_sha256=spec['mask_pixels_sha256'],mask_boundary_user_approved=True,
        motion='Rigid integer translation; same pixel offsets as prior control, now on cropped source',
        background_method='four_neighbor_boundary_color_propagation',
        occlusion_policy='Single object; no entry, exit, depth ordering or occlusion handling',
        limitations=['Source-derived keyframe, not independent synthetic imagery.',
            'Rigid translation does not demonstrate natural swimming.',
            'Binary mask preserves background mixtures within translucent fins.',
            'Background repair remains artifact-prone; annotations derive from the approved mask, not detections.'],frames=[])
    sheet=Image.new('RGB',(1280,290),'white');annotated=sheet.copy()
    maskbool=np.asarray(mask)>0;yy,xx=np.where(maskbool)
    for i,((frame,moved),(dx,dy)) in enumerate(zip(rendered,MOVES)):
        framepath=out/'frames'/('%06d.png'%i);frame.save(framepath)
        moved.save(out/'masks'/('%06d.png'%i))
        arr=np.asarray(frame);m=np.asarray(moved)>0
        assert np.array_equal(arr[yy+dy,xx+dx],np.asarray(original)[yy,xx])
        untouched=~(maskbool|m)
        assert np.array_equal(arr[untouched],np.asarray(original)[untouched])
        if i==0:assert np.array_equal(arr,np.asarray(original))
        revealed=maskbool&~m
        Image.fromarray(revealed.astype('uint8')*255).save(out/'diagnostics'/('revealed_%06d.png'%i))
        bbox=list(moved.getbbox())
        annotation=dict(frame_index=i,instance_id=1,class_id=4,class_name='fish',
            bbox_xyxy_pixels=bbox,bbox_xyxy=[v/256 for v in bbox],bbox_max_exclusive=True,
            provenance='Translated user-approved binary mask bounds; not independently verified ground truth',
            translation_pixels=[dx,dy])
        (out/'annotations'/('%06d.json'%i)).write_text(json.dumps(annotation,indent=2))
        report['frames'].append(dict(frame_index=i,translation_pixels=[dx,dy],sha256=sha(framepath),
            source_mask_pixels_preserved=True,unaffected_background_preserved=True,
            mask_pixels=int(m.sum()),filled_pixels_exposed=int(revealed.sum())))
        sheet.paste(frame,(256*i,34));ImageDraw.Draw(sheet).text((256*i+4,8),'Frame %d | dx=%d dy=%d'%(i,dx,dy),fill='black')
        view=frame.copy();draw=ImageDraw.Draw(view);draw.rectangle((bbox[0],bbox[1],bbox[2]-1,bbox[3]-1),outline='red')
        draw.text((bbox[0],max(0,bbox[1]-12)),'fish ID 1',fill='red')
        annotated.paste(view,(256*i,34));ImageDraw.Draw(annotated).text((256*i+4,8),'Mask-derived bounds | frame %d'%i,fill='black')
    sheet.save(out/'contact_sheet.png');annotated.save(out/'annotated_contact_sheet.png')
    assert {p:sha(ROOT/'baseline'/p) for p in before}==before
    assert sha(source)==spec['keyframe_sha256'] and sha(maskpath)==spec['mask_file_sha256']
    report['baseline_and_source_hashes_unchanged']=True
    (out/'qc.json').write_text(json.dumps(report,indent=2))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for f in out.rglob('*'):
            if f.is_file():bundle.write(f,str(f.relative_to(out)))
    print('Output:',str(out).encode('ascii','backslashreplace').decode())
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
