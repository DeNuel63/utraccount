from pathlib import Path
from typing import Sequence

from ulgf_baseline.layouts import read_yolo_layout

from .motion import MotionConfig, generate_object_sequences
from .schema import ClipManifest, FrameLayout, GenerationMetadata, MediumSpec, initial_objects_from_yolo
from .validation import validate_manifest


def build_layout_manifest(
    clip_id: str,
    source_image: Path,
    source_label: Path,
    classes: Sequence[str],
    width: int,
    height: int,
    seed: int,
    medium_description: str,
    motion: MotionConfig,
    config_name: str = "",
) -> ClipManifest:
    rows = read_yolo_layout(source_label, classes)
    initial_objects = initial_objects_from_yolo(rows, classes)
    sequences = generate_object_sequences(initial_objects, seed, motion)
    frames = tuple(
        FrameLayout(
            frame_index=index,
            image_path="frames/{:06d}.png".format(index),
            objects=objects,
        )
        for index, objects in enumerate(sequences)
    )
    manifest = ClipManifest(
        clip_id=clip_id,
        seed=seed,
        fps=motion.fps,
        width=width,
        height=height,
        medium=MediumSpec(description=medium_description),
        frames=frames,
        generation=GenerationMetadata(
            checkpoint="",
            config=config_name,
            temporal_mode="layout_only",
            source_image=str(source_image),
            motion={
                "model": "bounded_random_acceleration",
                "boundary_policy": motion.boundary_policy,
                "max_speed_box_fraction": motion.max_speed_box_fraction,
                "max_acceleration_box_fraction": motion.max_acceleration_box_fraction,
            },
        ),
    )
    validate_manifest(manifest)
    return manifest
