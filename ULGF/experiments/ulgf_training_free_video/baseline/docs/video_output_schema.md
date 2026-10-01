# ULGF Video Output Schema 1.0

## Scope decisions

Schema version 1.0 represents the initial video MVP:

- IDs are positive integers unique within one clip.
- IDs are assigned deterministically from source annotation order, starting at 1.
- One ID has exactly one `class_id` and `class_name` throughout a clip.
- The initial motion model has a fixed object count: no entry, exit, or full occlusion.
- Visibility is therefore `1.0` for every object in version 1.0 layout-only manifests.
- Boxes use normalized `xyxy` coordinates with eight decimal places in serialized JSON.
- Boxes that extend beyond the seed image are clipped during YOLO import; degenerate clipped boxes are rejected.
- Frame paths are relative POSIX paths under the clip directory.
- PNG frame sequences are canonical. Video files are optional previews.
- Dataset class IDs retain the class order supplied during import. A shared cross-dataset taxonomy is outside version 1.0.

## Required manifest fields

- `schema_version`: currently `1.0`.
- `clip_id`: non-empty external identifier.
- `seed`: integer controlling deterministic layout motion.
- `fps`: positive frame rate used to interpret motion.
- `width` and `height`: positive output dimensions.
- `medium.description`: clip-level water and lighting description.
- `frames`: ordered list with contiguous indices beginning at zero.
- `generation`: provenance and motion configuration.

Each frame contains a unique relative `image_path` and a list of objects. Each object contains `instance_id`, `class_id`, `class_name`, `bbox_xyxy`, and `visibility`.

## Compatibility

Readers accept additive fields within major version 1. A new major version is required for incompatible coordinate, identity, or frame-order semantics.

## Validation

Run:

```bash
python scripts/validate_video_clip.py path/to/manifest.json
```

After frame generation, require the image files as well:

```bash
python scripts/validate_video_clip.py path/to/manifest.json --require-images
```
