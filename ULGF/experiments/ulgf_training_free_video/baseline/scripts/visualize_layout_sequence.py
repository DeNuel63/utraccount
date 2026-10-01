#!/usr/bin/env python
"""Render layout-only frame previews from a ULGF clip manifest."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image, ImageDraw, ImageFont

from ulgf_video.schema import ClipManifest
from ulgf_video.validation import validate_manifest


PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def render(manifest, output_dir, background=None, overwrite=False):
    validate_manifest(manifest)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError("output directory is not empty; pass --overwrite to replace preview frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = None
    if background is not None:
        source = Image.open(str(background)).convert("RGB").resize((manifest.width, manifest.height))
    font = ImageFont.load_default()
    for frame in manifest.frames:
        canvas = source.copy() if source is not None else Image.new("RGB", (manifest.width, manifest.height), "#eaf4f7")
        draw = ImageDraw.Draw(canvas)
        for obj in frame.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            box = (
                round(x1 * manifest.width), round(y1 * manifest.height),
                round(x2 * manifest.width), round(y2 * manifest.height),
            )
            color = PALETTE[(obj.instance_id - 1) % len(PALETTE)]
            draw.rectangle(box, outline=color, width=3)
            label = "{} #{}".format(obj.class_name, obj.instance_id)
            text_box = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label, fill="white", font=font)
        target = output_dir / "{:06d}.png".format(frame.frame_index)
        temporary = target.with_name(target.stem + ".tmp.png")
        canvas.save(str(temporary))
        temporary.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render box-motion previews from a clip manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error("manifest does not exist: {}".format(args.manifest))
    if args.background is not None and not args.background.is_file():
        parser.error("background does not exist: {}".format(args.background))
    try:
        render(ClipManifest.read(args.manifest), args.output_dir, args.background, args.overwrite)
    except ValueError as error:
        parser.error(str(error))
    print("Rendered layout previews to {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
