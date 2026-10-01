"""Five controlled independent ULGF frames; no optimization or temporal claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'baseline'
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE))
from PIL import Image, ImageDraw
from ulgf_baseline.config import DEFAULT_RUOD_CLASSES
from ulgf_video.layout_sequence import build_layout_manifest
from ulgf_video.motion import MotionConfig
from ulgf_video.validation import validate_manifest

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def verify_snapshot():
    spec=json.loads((ROOT/'baseline_snapshot.json').read_text())
    for relative,expected in spec['files'].items():
        if digest(BASE/relative)!=expected:raise ValueError('Snapshot changed: '+relative)
    return digest(ROOT/'baseline_snapshot.json')

def safe_output(path, inputs=()):
    path=Path(path).resolve(); allowed=(ROOT/'runs').resolve()
    path.relative_to(allowed)
    if path==allowed:raise ValueError('Choose a named run beneath runs/')
    for source in inputs:
        source=Path(source).resolve()
        if path==source or path in source.parents or source in path.parents:
            raise ValueError('Output overlaps input')
    if path.exists():raise ValueError('Run already exists; choose a new name')
    return path

def layout(seed=0):
    return build_layout_manifest('five_frame_baseline',BASE/'assets/smoke/images/simple_012339.jpg',
        BASE/'assets/smoke/labels/simple_012339.txt',DEFAULT_RUOD_CLASSES,256,256,seed,
        'Fixed checkpoint prompt template and front camera',MotionConfig(frames=5),
        'controlled_independent_five_frames')

class ULGFAdapter:
    def __init__(self,checkpoint):
        import torch
        from accelerate.utils import set_seed
        from utils.generation_utils import load_checkpoint,bbox_encode
        if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
        self.torch=torch;self.set_seed=set_seed;self.encode=bbox_encode
        self.pipe,self.config=load_checkpoint(str(checkpoint))
        if self.config['dataset']!='ruod':raise ValueError('Expected RUOD checkpoint')
        for name in ('unet','vae','text_encoder','safety_checker'):
            module=getattr(self.pipe,name,None)
            if module is not None:module.eval();module.requires_grad_(False)
        if self.pipe.safety_checker is None:raise ValueError('Safety checker must remain enabled')
        self.pipe.to('cuda');self.pipe.enable_attention_slicing()

    def generate(self,frame,seed,steps,guidance):
        self.set_seed(seed)
        rows=[[o.class_name]+list(o.bbox_xyxy) for o in frame.objects]
        prompt=self.config['prompt_template'].format(camera='front',bbox=self.encode(rows,self.config))
        tokens=self.pipe.tokenizer(prompt,truncation=False)['input_ids']
        if len(tokens)>self.pipe.tokenizer.model_max_length:raise ValueError('Prompt would truncate')
        with self.torch.no_grad():
            result=self.pipe([prompt],guidance_scale=guidance,num_inference_steps=steps,height=256,width=256)
        if len(result.images)!=1:raise ValueError('Expected one image')
        flags=result.nsfw_content_detected
        return result.images[0],prompt,bool(flags and flags[0])

def checkpoint_record(checkpoint):
    required=('model_index.json','generation_config.json','unet/config.json',
        'unet/diffusion_pytorch_model.bin','vae/config.json','vae/diffusion_pytorch_model.bin',
        'text_encoder/config.json','text_encoder/pytorch_model.bin','scheduler/scheduler_config.json',
        'tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/added_tokens.json',
        'tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json',
        'safety_checker/config.json','safety_checker/pytorch_model.bin',
        'feature_extractor/preprocessor_config.json')
    for name in required:
        if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:
            raise ValueError('Missing checkpoint component: '+name)
    return {name:digest(checkpoint/name) for name in required}

def generate(manifest,output,adapter,seed=0,steps=100,guidance=5.0):
    output.mkdir(parents=True,exist_ok=False)
    (output/'frames').mkdir();(output/'annotations').mkdir()
    report=dict(status='RUNNING',training_started=False,temporal_mode='controlled_independent',
        annotation_status='requested_unverified',visual_quality='NOT_REVIEWED',frames=[])
    def save():
        temp=output/'run.pending.json';temp.write_text(json.dumps(report,indent=2));temp.replace(output/'run.json')
    save()
    try:
        for f in manifest.frames:
            image,prompt,flagged=adapter.generate(f,seed,steps,guidance)
            if image.size!=(256,256):raise ValueError('Incorrect output dimensions')
            path=output/f.image_path
            image.convert('RGB').save(path)
            objects=[dict(instance_id=o.instance_id,class_id=o.class_id,class_name=o.class_name,
                bbox_requested_xyxy=list(o.bbox_xyxy),bbox_verified_xyxy=None,
                verification_status='not_verified') for o in f.objects]
            (output/'annotations'/('{:06d}.json'.format(f.frame_index))).write_text(json.dumps(
                dict(frame_index=f.frame_index,objects=objects),indent=2))
            report['frames'].append(dict(frame_index=f.frame_index,prompt=prompt,seed=seed,
                safety_flagged=flagged,sha256=digest(path)))
            save();print('Generated frame {}/5'.format(f.frame_index+1),flush=True)
        manifest.write(output/'manifest.json')
        sheet=Image.new('RGB',(5*256,290),'white');draw=ImageDraw.Draw(sheet)
        for f in manifest.frames:
            with Image.open(output/f.image_path) as image:sheet.paste(image,(256*f.frame_index,30))
            draw.text((256*f.frame_index+5,5),'Frame {}'.format(f.frame_index),fill='black')
        sheet.save(output/'contact_sheet.png')
        report['status']='INFERENCE_COMPLETE_REVIEW_REQUIRED'
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);raise
    finally:save()
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--run-name',default='five_frame_001')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--guidance',type=float,default=5.0)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    import math
    if a.steps<1 or not math.isfinite(a.guidance) or a.guidance<0: p.error('Invalid inference settings')
    snapshot=verify_snapshot();m=layout(a.seed);validate_manifest(m)
    output=safe_output(ROOT/'runs'/a.run_name,[a.checkpoint] if a.checkpoint else [])
    if a.dry_run:
        print(json.dumps(dict(status='PLAN_PASS',frames=len(m.frames),snapshot_sha256=snapshot,
            model_loaded=False,training_started=False,output_written=False)))
        return
    if a.checkpoint is None:p.error('--checkpoint is required for generation')
    hashes=checkpoint_record(a.checkpoint)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    adapter=ULGFAdapter(a.checkpoint)  # Exactly one load for all five frames.
    m=replace(m,generation=replace(m.generation,checkpoint=str(a.checkpoint.resolve()),
                                   temporal_mode='controlled_independent'))
    report=generate(m,output,adapter,a.seed,a.steps,a.guidance)
    provenance=dict(snapshot_sha256=snapshot,checkpoint_sha256=hashes,
        checkpoint_path=str(a.checkpoint.resolve()),steps=a.steps,guidance=a.guidance,
        python=sys.version,torch=adapter.torch.__version__,cuda=adapter.torch.version.cuda,
        gpu=adapter.torch.cuda.get_device_name(0),source_image_sha256=digest(Path(m.generation.source_image)),
        safety_checker_disabled=False,training_started=False)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(report['status'])

if __name__=='__main__':main()
