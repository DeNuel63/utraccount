from pathlib import Path, PurePosixPath
from typing import Dict, List, Set, Tuple

from .schema import ClipManifest


SUPPORTED_SCHEMA_MAJOR = "1"


class ManifestValidationError(ValueError):
    pass


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ManifestValidationError("image_path must be a non-empty relative POSIX path: {}".format(value))


def validate_manifest(manifest: ClipManifest, clip_root: Path = None, require_images: bool = False) -> None:
    errors: List[str] = []
    if manifest.schema_version.split(".", 1)[0] != SUPPORTED_SCHEMA_MAJOR:
        errors.append("unsupported schema version {}".format(manifest.schema_version))
    if not manifest.clip_id.strip():
        errors.append("clip_id must not be empty")
    if manifest.fps <= 0:
        errors.append("fps must be positive")
    if manifest.width <= 0 or manifest.height <= 0:
        errors.append("width and height must be positive")
    if not manifest.frames:
        errors.append("at least one frame is required")

    expected_indices = list(range(len(manifest.frames)))
    actual_indices = [frame.frame_index for frame in manifest.frames]
    if actual_indices != expected_indices:
        errors.append("frame indices must be contiguous, ordered, and start at zero")

    identity_classes: Dict[int, Tuple[int, str]] = {}
    image_paths: Set[str] = set()
    for frame in manifest.frames:
        try:
            _validate_relative_path(frame.image_path)
        except ManifestValidationError as error:
            errors.append("frame {}: {}".format(frame.frame_index, error))
        if frame.image_path in image_paths:
            errors.append("frame {} reuses image_path {}".format(frame.frame_index, frame.image_path))
        image_paths.add(frame.image_path)
        frame_ids: Set[int] = set()
        for obj in frame.objects:
            if obj.instance_id <= 0:
                errors.append("frame {} has non-positive instance_id {}".format(frame.frame_index, obj.instance_id))
            if obj.instance_id in frame_ids:
                errors.append("frame {} repeats instance_id {}".format(frame.frame_index, obj.instance_id))
            frame_ids.add(obj.instance_id)
            if obj.class_id < 0 or not obj.class_name.strip():
                errors.append("frame {} instance {} has an invalid class".format(frame.frame_index, obj.instance_id))
            identity = (obj.class_id, obj.class_name)
            if obj.instance_id in identity_classes and identity_classes[obj.instance_id] != identity:
                errors.append("instance {} changes class across frames".format(obj.instance_id))
            identity_classes[obj.instance_id] = identity
            if len(obj.bbox_xyxy) != 4:
                errors.append("frame {} instance {} bbox must have four coordinates".format(frame.frame_index, obj.instance_id))
            else:
                x1, y1, x2, y2 = obj.bbox_xyxy
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    errors.append("frame {} instance {} has an invalid bbox".format(frame.frame_index, obj.instance_id))
            if not (0.0 <= obj.visibility <= 1.0):
                errors.append("frame {} instance {} visibility is outside [0, 1]".format(frame.frame_index, obj.instance_id))

        if require_images:
            if clip_root is None:
                errors.append("clip_root is required when require_images is true")
            elif not (clip_root / Path(frame.image_path)).is_file():
                errors.append("frame {} image does not exist: {}".format(frame.frame_index, frame.image_path))

    if errors:
        raise ManifestValidationError("Manifest validation failed:\n- " + "\n- ".join(errors))
