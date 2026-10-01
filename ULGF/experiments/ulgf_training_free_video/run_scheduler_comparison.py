"""Four GPU calls, one model load; scheduling A/B only, no image conditioning."""
import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from PIL import Image, ImageDraw
from run_five_frames import ROOT, verify_snapshot, safe_output, checkpoint_record, ULGFAdapter, layout, digest
from scheduler_control import corrected_class, corrected_source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if sys.version_info[:2]!=(3,7) or torch.__version__!='1.12.1+cu113' or not torch.cuda.is_available():
        raise RuntimeError('Restore validated isolated Python 3.7 / torch 1.12.1+cu113 GPU environment')
    output=safe_output(ROOT/'runs'/a.run_name,[a.checkpoint])
    snapshot=verify_snapshot()
    print('Hashing checkpoint components...',flush=True)
    hashes=checkpoint_record(a.checkpoint)
    if hashes['unet/diffusion_pytorch_model.bin']!='6f1ee83477953428b3cfc95769b7cfa1fdc5321b598b211192ecb25424f53bff':
        raise ValueError('Unexpected UNet hash')
    frame=layout(0).frames[0]
    frame=replace(frame,objects=(replace(frame.objects[0],bbox_xyxy=(.30,.27,.62,.62)),))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',training_started=False,source_image_conditioning=False,
                snapshot_sha256=snapshot,checkpoint_sha256=hashes,steps=100,guidance=5.0,
                python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),safety_checker_disabled=False,results=[])
    def save():
        temp=output/'report.pending.json';temp.write_text(json.dumps(report,indent=2));temp.replace(output/'report.json')
    save()
    try:
        print('Loading model once; two seeds x two scheduling paths.',flush=True)
        adapter=ULGFAdapter(a.checkpoint)
        import pipeline_custom.pipeline_prior as module
        original=type(adapter.pipe); corrected=corrected_class(module)
        pristine_scheduler=copy.deepcopy(adapter.pipe.scheduler)
        source=Path(module.__file__).read_text(encoding='utf-8')
        report['corrected_source_sha256']=hashlib.sha256(corrected_source(source).encode('utf-8')).hexdigest()
        sheet=Image.new('RGB',(512,584),'white')
        for row,seed in enumerate((3,4)):
            pair=[]
            for col,(label,cls) in enumerate((('original_skip5',original),('corrected_full',corrected))):
                adapter.pipe.__class__=cls
                adapter.pipe.scheduler=copy.deepcopy(pristine_scheduler)
                timesteps=[]
                hook=adapter.pipe.unet.register_forward_pre_hook(lambda model,args: timesteps.append(int(args[1].item())))
                try:
                    print('Seed {} / {}'.format(seed,label),flush=True)
                    image,prompt,flagged=adapter.generate(frame,seed,100,5.0)
                finally: hook.remove()
                expected=96 if col==0 else 101
                if len(timesteps)!=expected:raise ValueError('Unexpected denoising call count: '+str(timesteps))
                filename='seed{}_{}.png'.format(seed,label)
                image.convert('RGB').save(output/filename)
                result=dict(seed=seed,variant=label,prompt=prompt,image_path=filename,sha256=digest(output/filename),
                            safety_flagged=flagged,denoising_calls=len(timesteps),timesteps=timesteps,
                            requested_bbox_xyxy=list(frame.objects[0].bbox_xyxy),review_status='UNREVIEWED')
                pair.append(result);report['results'].append(result);save()
                shown=Image.new('RGB',(256,256),'gray') if flagged else image
                sheet.paste(shown,(256*col,row*292+36))
                ImageDraw.Draw(sheet).text((256*col+4,row*292+8),'seed {} {}{}'.format(seed,label,' FLAGGED' if flagged else ''),fill='black')
                sheet.save(output/'comparison.png')
            if pair[0]['prompt']!=pair[1]['prompt']:raise ValueError('A/B prompts differ')
        report['status']='SCHEDULER_AB_COMPLETE_VISUAL_REVIEW_REQUIRED'
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);raise
    finally: save()
    print(report['status']);print('Saved: '+str(output))

if __name__=='__main__':main()
