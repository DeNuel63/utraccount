"""Controlled repeated inference; no motion, model training or temporal correction."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from PIL import Image, ImageChops, ImageStat

import run_five_frames as runner
from ulgf_video.schema import ClipManifest
from ulgf_video.validation import validate_manifest


def fixed_layout(manifest):
    if len(manifest.frames)!=5:raise ValueError('Reference must have five frames')
    first=manifest.frames[0].objects
    return replace(manifest,clip_id='zero_motion_control',frames=tuple(
        replace(frame,objects=first) for frame in manifest.frames),
        generation=replace(manifest.generation,temporal_mode='zero_motion_control',
            motion={'model':'fixed_first_layout','max_speed_box_fraction':0,
                    'max_acceleration_box_fraction':0}))


def compare(paths):
    with Image.open(paths[0]) as source:reference=source.convert('RGB')
    rows=[]
    first_hash=runner.digest(paths[0])
    for index,path in enumerate(paths):
        with Image.open(path) as source:current=source.convert('RGB')
        if current.size!=reference.size:raise ValueError('Image dimensions differ')
        difference=ImageChops.difference(reference,current)
        maximum=max(high for low,high in difference.getextrema())
        rows.append(dict(frame_index=index,sha256=runner.digest(path),
            encoded_identical_to_frame0=runner.digest(path)==first_hash,
            pixels_identical_to_frame0=maximum==0,
            maximum_channel_difference=maximum,
            mean_absolute_channel_difference=sum(ImageStat.Stat(difference).mean)/3))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    a=p.parse_args()
    reference=a.reference_run.resolve();reference.relative_to((runner.ROOT/'runs').resolve())
    snapshot=runner.verify_snapshot()
    prior=json.loads((reference/'run.json').read_text())
    provenance=json.loads((reference/'provenance.json').read_text())
    if provenance['snapshot_sha256']!=snapshot:raise ValueError('Reference baseline differs')
    if prior['status']!='INFERENCE_COMPLETE_REVIEW_REQUIRED' or len(prior['frames'])!=5:
        raise ValueError('Reference run is incomplete')
    seeds={frame['seed'] for frame in prior['frames']}
    if len(seeds)!=1:raise ValueError('Reference used different seeds')
    seed=seeds.pop()
    original=ClipManifest.read(reference/'manifest.json')
    validate_manifest(original)
    for frame,record in zip(original.frames,prior['frames']):
        image=(reference/frame.image_path).resolve();image.relative_to(reference)
        if record['frame_index']!=frame.frame_index or runner.digest(image)!=record['sha256']:
            raise ValueError('Reference frame integrity mismatch')
    output=runner.safe_output(runner.ROOT/'runs'/a.run_name,[a.checkpoint,reference])
    print('Checking the same checkpoint; no model has been loaded yet.',flush=True)
    hashes=runner.checkpoint_record(a.checkpoint)
    if hashes!=provenance['checkpoint_sha256']:raise ValueError('Checkpoint differs from reference')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    manifest=fixed_layout(original);validate_manifest(manifest)
    print('Loading ULGF once for five identical-input calls.',flush=True)
    adapter=runner.ULGFAdapter(a.checkpoint)
    if adapter.torch.__version__!=provenance['torch'] or adapter.torch.version.cuda!=provenance['cuda']:
        raise ValueError('Torch/CUDA differs from reference')
    if adapter.torch.cuda.get_device_name(0)!=provenance['gpu']:
        raise ValueError('GPU differs from reference; use the same device for this control')
    result=runner.generate(manifest,output,adapter,seed,provenance['steps'],provenance['guidance'])
    result['temporal_mode']='zero_motion_control'
    prompts={row['prompt'] for row in result['frames']}
    prompt_match=len(prompts)==1 and next(iter(prompts))==prior['frames'][0]['prompt']
    comparisons=compare([output/frame.image_path for frame in manifest.frames])
    pixels_equal=all(row['pixels_identical_to_frame0'] for row in comparisons)
    flagged=any(row['safety_flagged'] for row in result['frames'])
    status=('CONTROL_INVALID_PROMPT_MISMATCH' if not prompt_match else
            'INCONCLUSIVE_SAFETY_FILTERED' if flagged else
            'IDENTICAL_INPUTS_IDENTICAL_PIXELS' if pixels_equal else
            'IDENTICAL_INPUTS_DIFFERENT_PIXELS')
    baseline_comparison=compare([reference/original.frames[0].image_path,output/manifest.frames[0].image_path])[1]
    report=dict(status=status,reference_run=str(reference),seed=seed,
        steps=provenance['steps'],guidance=provenance['guidance'],
        identical_layouts=True,identical_prompts=prompt_match,
        all_encoded_files_identical=all(r['encoded_identical_to_frame0'] for r in comparisons),
        all_pixels_identical=pixels_equal,safety_flags=[r['safety_flagged'] for r in result['frames']],
        comparisons=comparisons,comparison_to_original_frame0=baseline_comparison,
        training_started=False,limitation='This control does not validate anatomy, layout fidelity or video quality.')
    (output/'run.json').write_text(json.dumps(result,indent=2))
    (output/'zero_motion_comparison.json').write_text(json.dumps(report,indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(provenance,
        checkpoint_path=str(a.checkpoint.resolve()),reference_run=str(reference),
        control='zero_motion_five_calls',python=sys.version),indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
