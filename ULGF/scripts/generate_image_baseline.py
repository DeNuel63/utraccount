#!/usr/bin/env python
"""Safely reproduce the original ULGF still-image generation workflow."""

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ulgf_baseline.config import BaselineConfig, DEFAULT_RUOD_CLASSES
from ulgf_baseline.layouts import read_yolo_layout, to_yolo_lines
from ulgf_baseline.planning import PreflightError, build_generation_plan, describe_plan


def _parse_classes(value):
    candidate = Path(value)
    if candidate.is_file():
        classes = [line.strip() for line in candidate.read_text().splitlines() if line.strip()]
    else:
        classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("classes must be a comma-separated list or a non-empty text file")
    return tuple(classes)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate still-image ULGF baseline samples without modifying the source dataset."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="ULGF checkpoint directory")
    parser.add_argument("--image-dir", type=Path, required=True, help="Source image directory")
    parser.add_argument("--label-dir", type=Path, required=True, help="Source YOLO-label directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dedicated output directory")
    parser.add_argument(
        "--classes",
        type=_parse_classes,
        default=DEFAULT_RUOD_CLASSES,
        help="Comma-separated class names or a text file; defaults to RUOD classes",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, normally cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; image index is added deterministically")
    parser.add_argument("--seed-file", type=Path, help="Optional file containing one integer seed per image")
    parser.add_argument("--samples-per-layout", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing existing outputs")
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Explicitly disable the checkpoint safety checker",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without loading the model")
    return parser


def _atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(str(temporary), str(path))


def _atomic_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(str(temporary))
    os.replace(str(temporary), str(path))


def _run(config, items, pipeline_loader=None, seed_setter=None, bbox_encoder=None):
    if pipeline_loader is None or seed_setter is None or bbox_encoder is None:
        from accelerate.utils import set_seed
        from utils.generation_utils import bbox_encode, load_checkpoint

        pipeline_loader = pipeline_loader or load_checkpoint
        seed_setter = seed_setter or set_seed
        bbox_encoder = bbox_encoder or bbox_encode

    pipe, generation_config = pipeline_loader(str(config.checkpoint))
    pipe = pipe.to(config.device)
    if config.disable_safety_checker:
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    generation_config = dict(generation_config)
    generation_config["nsamples"] = config.samples_per_layout
    if config.guidance_scale is not None:
        generation_config["cfg_scale"] = config.guidance_scale
    if config.inference_steps is not None:
        generation_config["num_inference_steps"] = config.inference_steps

    run_metadata = {
        "checkpoint": str(config.checkpoint.resolve()),
        "device": config.device,
        "samples_per_layout": config.samples_per_layout,
        "guidance_scale": generation_config["cfg_scale"],
        "inference_steps": generation_config["num_inference_steps"],
        "classes": list(config.classes),
        "safety_checker_disabled": config.disable_safety_checker,
        "items": [],
    }

    for item in items:
        seed_setter(item.seed)
        objects = read_yolo_layout(item.source_label, config.classes)
        prompt_layout = {"camera": "front", "bbox": [list(obj) for obj in objects]}
        encoded_layout = dict(prompt_layout)
        encoded_layout["bbox"] = bbox_encoder(encoded_layout["bbox"], generation_config)
        prompt = generation_config["prompt_template"].format(**encoded_layout)
        result = pipe(
            config.samples_per_layout * [prompt],
            guidance_scale=generation_config["cfg_scale"],
            num_inference_steps=generation_config["num_inference_steps"],
            height=int(generation_config["height"]),
            width=int(generation_config["width"]),
        )
        if len(result.images) != len(item.output_images):
            raise RuntimeError(
                "Pipeline returned {} images; expected {}".format(len(result.images), len(item.output_images))
            )
        for image, output_path in zip(result.images, item.output_images):
            _atomic_image(image.convert("RGB"), output_path)
        _atomic_text(item.output_label, "\n".join(to_yolo_lines(objects, config.classes)) + "\n")
        run_metadata["items"].append(
            {
                "source_image": str(item.source_image.resolve()),
                "source_label": str(item.source_label.resolve()),
                "seed": item.seed,
                "outputs": [str(path.relative_to(config.output_dir)) for path in item.output_images],
                "label": str(item.output_label.relative_to(config.output_dir)),
                "prompt": prompt,
            }
        )

    _atomic_text(config.output_dir / "run.json", json.dumps(run_metadata, indent=2) + "\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BaselineConfig(
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        device=args.device,
        seed=args.seed,
        seed_file=args.seed_file,
        samples_per_layout=args.samples_per_layout,
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        overwrite=args.overwrite,
        disable_safety_checker=args.disable_safety_checker,
    )
    try:
        items = build_generation_plan(config)
    except PreflightError as error:
        parser.error(str(error))
    print(describe_plan(config, items))
    if args.dry_run:
        print("Dry run complete; no output was written and no model was loaded.")
        return 0
    _run(config, items)
    print("Generated {} layout(s) in {}".format(len(items), config.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
