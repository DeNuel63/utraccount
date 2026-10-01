"""One source, VAE reconstruction, three stock Diffusers 0.4.1 img2img calls."""
import argparse, copy, inspect, json, os, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from run_five_frames import ROOT, BASE, verify_snapshot, safe_output, checkpoint_record, ULGFAdapter, layout, digest
from image_conditioning import prepare, prepare_crop, normalized

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--fish-crop',action='store_true',help='Square source crop, no padding; strengths .05 and .10 only')
    a=p.parse_args()
    strengths=(.05,.10) if a.fish_crop else (.15,.25,.35)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch, diffusers
    from diffusers import StableDiffusionImg2ImgPipeline
    if sys.version_info[:2]!=(3,7) or torch.__version__!='1.12.1+cu113' or diffusers.__version__!='0.4.1' or not torch.cuda.is_available():
        raise RuntimeError('Restore validated isolated Python 3.7 / torch 1.12.1+cu113 / diffusers 0.4.1 GPU environment')
    if 'init_image' not in inspect.signature(StableDiffusionImg2ImgPipeline.__call__).parameters:
        raise RuntimeError('Unexpected img2img API')
    output=safe_output(ROOT/'runs'/a.run_name,[a.checkpoint]);snapshot=verify_snapshot()
    print('Verifying checkpoint hashes...',flush=True);hashes=checkpoint_record(a.checkpoint)
    if hashes['unet/diffusion_pytorch_model.bin']!='6f1ee83477953428b3cfc95769b7cfa1fdc5321b598b211192ecb25424f53bff':raise ValueError('UNet hash mismatch')
    source=BASE/'assets/smoke/images/simple_012339.jpg';frame=layout(0).frames[0]
    with Image.open(source) as image:
        prepared,box,geometry=(prepare_crop if a.fish_crop else prepare)(image,frame.objects[0].bbox_xyxy)
    frame=replace(frame,objects=(replace(frame.objects[0],bbox_xyxy=box),))
    output.mkdir(parents=True,exist_ok=False)
    prepared.save(output/'source_prepared.png')
    report=dict(status='RUNNING',training_started=False,dataset_accepted=False,visual_quality='UNREVIEWED',
        source_image_conditioning=True,source_path=str(source),source_sha256=digest(source),
        source_prepared_sha256=digest(output/'source_prepared.png'),geometry=geometry,
        requested_bbox_xyxy=box,snapshot_sha256=snapshot,checkpoint_sha256=hashes,
        seed=4,guidance=5.0,nominal_steps=100,strengths=list(strengths),safety_checker_disabled=False,
        method='stock_diffusers_img2img_with_ULGF_components',
        python=sys.version,torch=torch.__version__,diffusers=diffusers.__version__,cuda=torch.version.cuda,
        limitations=['Source-derived diagnostic, not independent synthetic data or leakage-cleared evaluation.',
                    'Crop removes scene context; boxes remain requested, not verified.' if a.fish_crop else
                    'Letterbox padding may influence generated content; boxes remain requested, not verified.'],results=[])
    def save():
        t=output/'report.pending.json';t.write_text(json.dumps(report,indent=2));t.replace(output/'report.json')
    save()
    try:
        print('Loading model once; sharing frozen components with stock img2img.',flush=True)
        adapter=ULGFAdapter(a.checkpoint);base=adapter.pipe
        pipe=StableDiffusionImg2ImgPipeline(**{n:getattr(base,n) for n in
             ('vae','text_encoder','tokenizer','unet','scheduler','safety_checker','feature_extractor')})
        pipe.to('cuda');pipe.enable_attention_slicing()
        pristine=copy.deepcopy(pipe.scheduler)
        rows=[[o.class_name]+list(o.bbox_xyxy) for o in frame.objects]
        adapter.set_seed(4)
        prompt=adapter.config['prompt_template'].format(camera='front',bbox=adapter.encode(rows,adapter.config))
        if len(pipe.tokenizer(prompt,truncation=False)['input_ids'])>pipe.tokenizer.model_max_length:raise ValueError('Prompt truncation')
        report['prompt']=prompt
        tensor=torch.from_numpy(normalized(prepared)).to(device='cuda',dtype=next(pipe.vae.parameters()).dtype)
        sheet=Image.new('RGB',((2+len(strengths))*256,288),'white')
        def record(name,image,flag,extra):
            if image.size!=(256,256):raise ValueError('Unexpected dimensions')
            image.save(output/(name+'.png'))
            item=dict(name=name,sha256=digest(output/(name+'.png')),safety_flagged=flag,**extra)
            report['results'].append(item);save()
            col=len(report['results'])
            shown=Image.new('RGB',(256,256),'gray') if flag else image
            sheet.paste(shown,(col*256,32));ImageDraw.Draw(sheet).text((col*256+4,8),name+(' FLAGGED' if flag else ''),fill='black')
            sheet.save(output/'comparison.png')
        sheet.paste(prepared,(0,32));ImageDraw.Draw(sheet).text((4,8),'Source crop' if a.fish_crop else 'Source (letterboxed)',fill='black')
        # Legacy posterior sampling can promote half latents to float32.
        # Autocast also covers this promotion inside the stock img2img pipeline.
        report['precision_policy']='fp16_weights_cuda_autocast_explicit_reconstruction_cast_v2'
        with torch.no_grad(), torch.cuda.amp.autocast():
            adapter.set_seed(4)
            z=pipe.vae.encode(tensor).latent_dist.sample()
            report['vae_sample_dtype']=str(z.dtype)
            decoder_weight=pipe.vae.post_quant_conv.weight
            decode_input=((z*.18215)/.18215).to(device=decoder_weight.device,dtype=decoder_weight.dtype)
            report['vae_decode_input_dtype']=str(decode_input.dtype)
            report['vae_decoder_weight_dtype']=str(decoder_weight.dtype)
            if not torch.isfinite(decode_input).all():raise ValueError('Non-finite VAE latent')
            save()
            decoded=pipe.vae.decode(decode_input).sample
            if not torch.isfinite(decoded).all():raise ValueError('Non-finite VAE reconstruction')
            arr=(decoded/2+.5).clamp(0,1).float().cpu().permute(0,2,3,1).numpy()
            features=pipe.feature_extractor(pipe.numpy_to_pil(arr),return_tensors='pt').to('cuda')
            arr,flags=pipe.safety_checker(images=arr,clip_input=features.pixel_values.to(tensor.dtype))
            record('VAE_reconstruction',pipe.numpy_to_pil(arr)[0],bool(flags[0]),dict(denoising_calls=0,posterior='sample_seed4'))
            for strength in strengths:
                print('Generating strength '+str(strength),flush=True)
                pipe.scheduler=copy.deepcopy(pristine);adapter.set_seed(4);timesteps=[]
                hook=pipe.unet.register_forward_pre_hook(lambda module,args:timesteps.append(int(args[1].item())))
                try:
                    result=pipe(prompt=[prompt],init_image=tensor.clone(),strength=strength,
                                num_inference_steps=100,guidance_scale=5.0)
                finally:hook.remove()
                record('strength_'+str(strength),result.images[0],bool(result.nsfw_content_detected[0]),
                       dict(strength=strength,timesteps=timesteps,denoising_calls=len(timesteps)))
        report['status']='IMAGE_CONDITIONING_COMPLETE_REVIEW_REQUIRED'
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);raise
    finally:save()
    print(report['status']);print('Saved: '+str(output))

if __name__=='__main__':main()
