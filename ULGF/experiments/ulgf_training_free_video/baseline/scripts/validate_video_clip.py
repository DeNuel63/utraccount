#!/usr/bin/env python
"""Validate a ULGF video manifest and optionally its rendered frame files."""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ulgf_video.schema import ClipManifest
from ulgf_video.validation import ManifestValidationError, validate_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate a ULGF clip manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error("manifest does not exist: {}".format(args.manifest))
    try:
        manifest = ClipManifest.read(args.manifest)
        validate_manifest(manifest, args.manifest.parent, args.require_images)
    except (KeyError, TypeError, ValueError, ManifestValidationError) as error:
        parser.error(str(error))
    print("Manifest is valid: {} frames, {} clip-local identities".format(
        len(manifest.frames), len({obj.instance_id for frame in manifest.frames for obj in frame.objects})
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
