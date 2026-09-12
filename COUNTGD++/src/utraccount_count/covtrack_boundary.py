"""Explicit, dependency-free preparation of CountGD++ detections for COVTrack.

COVTrack internally represents a detection as ``[left, top, right, bottom,
score]`` plus an integer class label.  This module only prepares that geometric
and semantic input.  It deliberately does not claim to run COVTrack: the
current COVTrack tracker also requires RoI-derived appearance embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import DetectionBatch


class COVTrackCompatibilityError(ValueError):
    """Raised when an exported CountGD++ batch cannot safely enter COVTrack."""


@dataclass(frozen=True)
class COVTrackDetections:
    """One frame in COVTrack's detector representation, without embeddings."""

    sequence_id: str
    frame_index: int
    boxes_xyxy_score: tuple[tuple[float, float, float, float, float], ...]
    label_ids: tuple[int, ...]


def to_covtrack_detections(
    batch: DetectionBatch, class_to_index: Mapping[str, int]
) -> COVTrackDetections:
    """Convert one validated batch to COVTrack's pixel ``xyxy + score`` form.

    ``class_to_index`` must be the same class vocabulary and index ordering as
    COVTrack's configured ``roi_head.CLASSES``.  The returned values can be
    converted to the framework tensors COVTrack expects at its external-
    detection injection point.
    """
    boxes: list[tuple[float, float, float, float, float]] = []
    label_ids: list[int] = []
    for detection in batch.detections:
        if detection.label not in class_to_index:
            raise COVTrackCompatibilityError(
                f"No COVTrack class index was supplied for label {detection.label!r}."
            )
        label_id = class_to_index[detection.label]
        if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id < 0:
            raise COVTrackCompatibilityError(
                f"COVTrack class index for {detection.label!r} must be a non-negative integer."
            )
        centre_x, centre_y, width, height = detection.bbox_xywh_center
        boxes.append(
            (
                centre_x - width / 2,
                centre_y - height / 2,
                centre_x + width / 2,
                centre_y + height / 2,
                detection.confidence,
            )
        )
        label_ids.append(label_id)
    return COVTrackDetections(
        sequence_id=batch.sequence_id,
        frame_index=batch.frame_index,
        boxes_xyxy_score=tuple(boxes),
        label_ids=tuple(label_ids),
    )


def validate_covtrack_sequence(batches: Iterable[DetectionBatch]) -> tuple[DetectionBatch, ...]:
    """Require the frame order COVTrack uses to initialise and maintain state."""
    sequence = tuple(batches)
    if not sequence:
        raise COVTrackCompatibilityError("COVTrack requires at least one frame.")
    sequence_ids = {batch.sequence_id for batch in sequence}
    if len(sequence_ids) != 1:
        raise COVTrackCompatibilityError("A COVTrack run must contain exactly one sequence.")
    frame_indices = tuple(batch.frame_index for batch in sequence)
    expected = tuple(range(len(sequence)))
    if frame_indices != expected:
        raise COVTrackCompatibilityError(
            "COVTrack frames must be ordered consecutively from frame_index 0."
        )
    return sequence
