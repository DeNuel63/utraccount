"""CPU single-object translation control. Not a photorealistic video generator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fill_hole(image, mask):
    """Boundary-color propagation, deliberately not presented as scene recovery."""
    values=np.asarray(image,dtype=np.float32).copy()
    missing=np.asarray(mask)>0
    known=~missing
    if not known.any():raise ValueError('Mask covers the entire image')
    values[missing]=0
    for _ in range(sum(missing.shape)):
        if known.all():break
        sums=np.zeros_like(values);counts=np.zeros(known.shape,dtype=np.float32)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            shifted=np.roll(known,(dy,dx),(0,1))
            if dy==-1:shifted[-1,:]=False
            if dy==1:shifted[0,:]=False
            if dx==-1:shifted[:,-1]=False
            if dx==1:shifted[:,0]=False
            sums+=np.roll(values,(dy,dx),(0,1))*shifted[:,:,None]
            counts+=shifted
        frontier=(~known)&(counts>0)
        if not frontier.any():raise ValueError('Background fill stalled')
        values[frontier]=sums[frontier]/counts[frontier,None]
        known[frontier]=True
    if not known.all():raise ValueError('Background fill incomplete')
    return Image.fromarray(np.rint(values).clip(0,255).astype(np.uint8))


def translated(image,background,mask,dx,dy):
    if dx==0 and dy==0:return image.copy(),mask.copy()
    box=mask.getbbox()
    if box is None:raise ValueError('Empty object mask')
    x1,y1,x2,y2=box
    if x1+dx<0 or y1+dy<0 or x2+dx>image.width or y2+dy>image.height:
        raise ValueError('Translation would clip the animal; rejected')
    result=background.copy()
    result.paste(image.crop(box),(x1+dx,y1+dy),mask.crop(box))
    moved=Image.new('L',image.size,0)
    moved.paste(mask.crop(box),(x1+dx,y1+dy))
    return result,moved


def offsets(frames,width,height):
    first=frames[0]['objects']
    if len(first)!=1:raise ValueError('This bounded control supports exactly one object; occlusion is unsupported')
    original=first[0];box=original['bbox_xyxy']
    if not all(math.isfinite(v) for v in box):raise ValueError('Non-finite box')
    cx=(box[0]+box[2])*width/2;cy=(box[1]+box[3])*height/2
    result=[]
    for i,f in enumerate(frames):
        if f['frame_index']!=i or len(f['objects'])!=1:raise ValueError('Invalid frame/object count')
        o=f['objects'][0];b=o['bbox_xyxy']
        if (o['instance_id'],o['class_id'],o['class_name'])!=(original['instance_id'],original['class_id'],original['class_name']):
            raise ValueError('Identity/class changed')
        if not all(math.isfinite(v) for v in b) or not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):
            raise ValueError('Invalid box')
        if abs((b[2]-b[0])-(box[2]-box[0]))>1e-6 or abs((b[3]-b[1])-(box[3]-box[1]))>1e-6:
            raise ValueError('Scaling unsupported; this control preserves source pixels')
        dx=round((b[0]+b[2])*width/2-cx);dy=round((b[1]+b[3])*height/2-cy)
        if abs(dx)>width*.25 or abs(dy)>height*.25:raise ValueError('Motion exceeds bounded control range')
        result.append((dx,dy))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--mask',type=Path,help='Optional reviewed binary PNG at the original frame dimensions')
    p.add_argument('--approval',type=Path,help='Optional hash-bound human approval record')
    a=p.parse_args();root=Path(__file__).resolve().parent
    reference=a.reference_run.resolve();reference.relative_to((root/'runs').resolve())
    output=(root/'runs'/a.run_name).resolve();output.relative_to((root/'runs').resolve())
    if output==root/'runs' or output.exists():raise ValueError('Choose a new run name')
    if reference in output.parents or output in reference.parents:
        raise ValueError('Output must not overlap the reference run')
    m=json.loads((reference/'manifest.json').read_text());log=json.loads((reference/'run.json').read_text())
    frames=m['frames']
    if len(frames)!=5:raise ValueError('Expected five reference frames')
    records={r['frame_index']:r for r in log['frames']}
    originals=[];hashes={}
    for f in frames:
        path=(reference/f['image_path']).resolve();path.relative_to(reference)
        h=sha(path)
        if h!=records[f['frame_index']]['sha256']:raise ValueError('Reference hash mismatch')
        if records[f['frame_index']].get('safety_flagged'):raise ValueError('Safety-filtered reference rejected')
        hashes[f['frame_index']]=h
        with Image.open(path) as image:
            if image.size!=(m['width'],m['height']):raise ValueError('Unexpected frame size')
            originals.append(image.convert('RGB'))
    source=originals[0];moves=offsets(frames,*source.size)
    b=frames[0]['objects'][0]['bbox_xyxy']
    if a.mask:
        with Image.open(a.mask) as image:mask=image.convert('L')
        if mask.size!=source.size or not set(np.unique(np.asarray(mask))).issubset({0,255}):
            raise ValueError('Mask must be binary, matching frame dimensions')
        mask_method='provided_binary_mask'
    else:
        mask=Image.new('L',source.size,0)
        # Rectangular region deliberately includes context; not object segmentation.
        box=(max(0,math.floor(b[0]*source.width)-4),max(0,math.floor(b[1]*source.height)-4),
             min(source.width-1,math.ceil(b[2]*source.width)+4),min(source.height-1,math.ceil(b[3]*source.height)+4))
        ImageDraw.Draw(mask).rectangle(box,fill=255)
        mask_method='requested_box_plus_4px_context_DIAGNOSTIC_ONLY'
    if mask.getbbox() is None:raise ValueError('Empty mask')
    mask_digest=hashlib.sha256(mask.tobytes()).hexdigest()
    approved=False
    if a.approval:
        approval=json.loads(a.approval.read_text())
        if approval['keyframe_sha256']!=hashes[0] or approval['mask_pixels_sha256']!=mask_digest:
            raise ValueError('Approval does not match this keyframe and mask')
        approved=(a.mask is not None and approval.get('animal_resemblance_approved') is True
                  and approval.get('mask_covers_complete_animal') is True
                  and bool(approval.get('reviewer')))
        if not approved:raise ValueError('Human keyframe/mask approval is incomplete')
    background=fill_hole(source,mask)
    # Check every move before creating output, including clipping and mask extent.
    generated=[translated(source,background,mask,*move) for move in moves]
    output.mkdir(parents=True,exist_ok=False)
    for name in ('frames','masks','annotations','diagnostics'):(output/name).mkdir()
    mask.save(output/'diagnostics/source_mask.png');background.save(output/'diagnostics/background_fill.png')
    overlay=Image.blend(source,Image.composite(Image.new('RGB',source.size,'red'),source,mask),.4)
    overlay.save(output/'diagnostics/source_mask_overlay.png')
    report=dict(status='DIAGNOSTIC_ONLY_REVIEW_REQUIRED',reference=str(reference),
        reference_manifest_sha256=sha(reference/'manifest.json'),reference_frame_sha256=hashes,
        method='single_keyframe_integer_translation',animal_resemblance_approved=approved,
        mask_method=mask_method,mask_pixels_sha256=mask_digest,keyframe_sha256=hashes[0],
        background_method='four_neighbor_boundary_color_propagation',
        occlusion_policy='single_object_no_entry_exit_or_occlusion',
        training_started=False,model_loaded=False,frames=[],dataset_accepted=False,
        limitations=['Warping cannot repair malformed anatomy or unnatural source color.',
                    'Background fill is an artifact-prone diagnostic, not recovered scene geometry.',
                    'Mask and intended boxes are not verified tracking ground truth.'])
    panel=Image.new('RGB',(5*source.width,2*source.height+100),'white');draw=ImageDraw.Draw(panel)
    draw.text((8,5),'Independent generation (top) / Single-keyframe warp control (bottom)',fill='black')
    mask_bool=np.asarray(mask)>0
    for i,((image,moved),(dx,dy)) in enumerate(zip(generated,moves)):
        image.save(output/'frames'/('{:06d}.png'.format(i)));moved.save(output/'masks'/('{:06d}.png'.format(i)))
        source_pixels=np.asarray(source)[mask_bool]
        # Reverse-index exactly the source mask coordinates; no interpolation.
        yy,xx=np.where(mask_bool);dest_pixels=np.asarray(image)[yy+dy,xx+dx]
        equal=bool(np.array_equal(source_pixels,dest_pixels))
        if not equal:raise RuntimeError('Translation unexpectedly altered foreground pixels')
        revealed=np.asarray(mask_bool & ~(np.asarray(moved)>0),dtype=np.uint8)*255
        Image.fromarray(revealed).save(output/'diagnostics'/('revealed_hole_%06d.png'%i))
        obj=frames[i]['objects'][0]
        actual=[b[0]+dx/source.width,b[1]+dy/source.height,b[2]+dx/source.width,b[3]+dy/source.height]
        annotation=dict(frame_index=i,instance_id=obj['instance_id'],class_name=obj['class_name'],
            class_id=obj['class_id'],bbox_requested_xyxy=obj['bbox_xyxy'],
            bbox_transformed_source_layout_xyxy=actual,bbox_verified_xyxy=None,
            verification_status='not_verified',translation_pixels=[dx,dy])
        (output/'annotations'/('%06d.json'%i)).write_text(json.dumps(annotation,indent=2))
        report['frames'].append(dict(frame_index=i,translation_pixels=[dx,dy],
            source_mask_pixels_preserved=equal,filled_pixels_exposed=int((revealed>0).sum())))
        panel.paste(originals[i],(i*source.width,35));panel.paste(image,(i*source.width,source.height+80))
        draw.text((i*source.width+5,20),'Frame %d'%i,fill='black')
        draw.text((i*source.width+5,source.height+60),'Warp dx=%d dy=%d'%(dx,dy),fill='black')
    panel.save(output/'comparison.png')
    (output/'qc.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Saved:',output)

if __name__=='__main__':main()
