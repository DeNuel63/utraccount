"""Paste this entire file into ONE cell in the isolated inference notebook."""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import subprocess
from IPython.display import display, Image as DisplayImage

root = Path('/content/drive/MyDrive/ulgf_training_free_video')
python = Path('/content/ulgf-video-py37/bin/python')
checkpoint = Path('/content/drive/MyDrive/ULGF-assets/checkpoints/ruod_256_bz16/checkpoint')
for required in (python, root/'run_five_frames.py', root/'baseline_snapshot.json', checkpoint):
    if not required.exists():
        raise FileNotFoundError('Restore the isolated inference environment/input: ' + str(required))
name = 'keyframe_candidates_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

script = r'''
import os, sys, json
from pathlib import Path
from dataclasses import replace, asdict
from PIL import Image, ImageDraw
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
from run_five_frames import (ROOT, verify_snapshot, safe_output, checkpoint_record,
                             ULGFAdapter, layout, digest)
import torch
if sys.version_info[:2] != (3,7) or torch.__version__ != '1.12.1+cu113':
    raise RuntimeError('Use the validated isolated Python 3.7 / torch 1.12.1+cu113 environment')
if not torch.cuda.is_available():
    raise RuntimeError('GPU required for this bounded candidate generation')
checkpoint=Path(sys.argv[1]); name=sys.argv[2]
output=safe_output(ROOT/'runs'/name,[checkpoint])
print('Verifying frozen baseline and checkpoint hashes; this can take several minutes.',flush=True)
snapshot=verify_snapshot()
hashes=checkpoint_record(checkpoint)
expected='6f1ee83477953428b3cfc95769b7cfa1fdc5321b598b211192ecb25424f53bff'
if hashes['unet/diffusion_pytorch_model.bin']!=expected:
    raise ValueError('UNet reference hash mismatch')
base=layout(0).frames[0]
if len(base.objects)!=1 or base.objects[0].class_name!='fish':
    raise ValueError('Expected a single fish fixture')
layouts=[('wide',(0.25,0.35,0.65,0.60)),
         ('near_square',(0.30,0.27,0.62,0.62))]
report=dict(status='RUNNING',training_started=False,safety_checker_disabled=False,
    animal_resemblance_approved=False,dataset_accepted=False,
    purpose='Bounded exploratory keyframe selection, not an evaluation or temporal test',
    snapshot_sha256=snapshot,checkpoint_path=str(checkpoint),checkpoint_sha256=hashes,
    python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),steps=100,guidance=5.0,candidates=[])
output.mkdir(parents=True,exist_ok=False)
(output/'frames').mkdir()
def save():
    temp=output/'report.pending.json'
    temp.write_text(json.dumps(report,indent=2))
    temp.replace(output/'report.json')
save()
try:
    print('Loading model ONCE. Six images maximum; no optimizer or training.',flush=True)
    adapter=ULGFAdapter(checkpoint)
    sheet=Image.new('RGB',(3*256,2*292),'white')
    overlay_sheet=sheet.copy()
    for row,(label,box) in enumerate(layouts):
        frame=replace(base,objects=(replace(base.objects[0],bbox_xyxy=box),))
        for col,seed in enumerate((3,4,5)):
            candidate='{}_seed{}'.format(label,seed)
            print('Generating '+candidate,flush=True)
            image,prompt,flagged=adapter.generate(frame,seed,100,5.0)
            if image.size!=(256,256):raise ValueError('Unexpected image dimensions')
            image=image.convert('RGB')
            path=output/'frames'/(candidate+'.png');image.save(path)
            report['candidates'].append(dict(candidate_id=candidate,image_path=str(path.relative_to(output)),
                sha256=digest(path),seed=seed,prompt=prompt,requested_object=asdict(frame.objects[0]),
                safety_flagged=flagged,review_status='REJECT_SAFETY_FLAG' if flagged else 'UNREVIEWED'))
            save()
            # Never present flagged output as a selectable candidate.
            shown=Image.new('RGB',(256,256),'gray') if flagged else image
            annotated=shown.copy()
            if not flagged:
                ImageDraw.Draw(annotated).rectangle(tuple(round(v*256) for v in box),outline='red',width=2)
            title=candidate+(' FLAGGED' if flagged else '')
            for canvas,tile in ((sheet,shown),(overlay_sheet,annotated)):
                canvas.paste(tile,(col*256,row*292+36))
                ImageDraw.Draw(canvas).text((col*256+5,row*292+8),title,fill='black')
            sheet.save(output/'contact_sheet.png')
            overlay_sheet.save(output/'requested_boxes.png')
    report['status']='CANDIDATES_COMPLETE_VISUAL_REVIEW_REQUIRED'
except Exception as error:
    report['status']='FAILED';report['error']=str(error)
    raise
finally:
    save()
print('Saved: '+str(output),flush=True)
print(report['status'],flush=True)
'''

print('GPU inference: six candidates, one model load. Normal training files remain untouched.', flush=True)
# Stream progress instead of hiding checkpoint verification and inference output.
process = subprocess.Popen([str(python), '-u', '-c', script, str(checkpoint), name],
                           cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
finally:
    if process.poll() is None:
        process.terminate()
        process.wait()
if code:
    raise RuntimeError('Candidate generation failed; preserve the run and send the error. No automatic retry.')
output = root/'runs'/name
display(DisplayImage(filename=str(output/'contact_sheet.png')))
display(DisplayImage(filename=str(output/'requested_boxes.png')))
print('Send both sheets for review. No keyframe has been approved and no masking/warping was run.')
