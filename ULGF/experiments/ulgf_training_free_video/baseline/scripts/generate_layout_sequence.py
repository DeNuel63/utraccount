#!/usr/bin/env python
"""Create and validate a deterministic ULGF video layout manifest."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.generate_image_baseline import _parse_classes
from ulgf_baseline.config import DEFAULT_RUOD_CLASSES
from ulgf_video.layout_sequence import build_layout_manifest
from ulgf_video.motion import MotionConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a layout-only synthetic clip manifest.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--classes", type=_parse_classes, default=DEFAULT_RUOD_CLASSES)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--medium", default="underwater scene")
    parser.add_argument("--max-speed-box-fraction", type=float, default=0.15)
    parser.add_argument("--max-acceleration-box-fraction", type=float, default=0.02)
    parser.add_argument("--boundary-policy", choices=("reflect", "clamp"), default="reflect")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.source_image.is_file():
        parser.error("source image does not exist: {}".format(args.source_image))
    if not args.source_label.is_file():
        parser.error("source label does not exist: {}".format(args.source_label))
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it: {}".format(args.output))
    motion = MotionConfig(
        frames=args.frames,
        fps=args.fps,
        max_speed_box_fraction=args.max_speed_box_fraction,
        max_acceleration_box_fraction=args.max_acceleration_box_fraction,
        boundary_policy=args.boundary_policy,
    )
    try:
        manifest = build_layout_manifest(
            clip_id=args.clip_id,
            source_image=args.source_image,
            source_label=args.source_label,
            classes=args.classes,
            width=args.width,
            height=args.height,
            seed=args.seed,
            medium_description=args.medium,
            motion=motion,
            config_name="command-line",
        )
    except ValueError as error:
        parser.error(str(error))
    manifest.write(args.output)
    print("Wrote {} frames to {}".format(len(manifest.frames), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
