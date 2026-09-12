"""Stable, serialisable contracts shared at the CountGD++/COVTrack boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

SCHEMA_VERSION = "utraccount.detection-batch/v1"


@dataclass(frozen=True)
class FrameInfo:
    uri: str
    width_px: int
    height_px: int
    timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.uri or self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("Frame uri and positive pixel dimensions are required.")


@dataclass(frozen=True)
class Detection:
    detection_id: str
    bbox_xywh_center: tuple[float, float, float, float]
    label: str
    confidence: float
    ground_truth_instance_id: str | None = None

    def __post_init__(self) -> None:
        x, y, width, height = self.bbox_xywh_center
        if not self.detection_id or not self.label:
            raise ValueError("Detection id and label are required.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1.")
        if width <= 0 or height <= 0:
            raise ValueError("Detection width and height must be positive.")
        if x < 0 or y < 0:
            raise ValueError("Detection centre coordinates cannot be negative.")
        if not all(isfinite(value) for value in self.bbox_xywh_center):
            raise ValueError("Detection coordinates must be finite numbers.")


@dataclass(frozen=True)
class InferenceSource:
    model_version: str
    prompt: str
    confidence_threshold: float
    raw_output_uri: str | None = None
    model_name: str = "CountGD++"
    box_format: str = "xywh_center"
    coordinate_space: str = "pixel"

    def __post_init__(self) -> None:
        if self.model_name != "CountGD++":
            raise ValueError("This contract is reserved for CountGD++ outputs.")
        if not self.model_version:
            raise ValueError("A model version or checkpoint identifier is required.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")


@dataclass(frozen=True)
class DetectionBatch:
    sequence_id: str
    frame_index: int
    frame: FrameInfo
    source: InferenceSource
    detections: tuple[Detection, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sequence_id or self.frame_index < 0:
            raise ValueError("Sequence id and a non-negative frame index are required.")
        ids = [d.detection_id for d in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("Detection IDs must be unique within a frame.")
        for detection in self.detections:
            centre_x, centre_y, width, height = detection.bbox_xywh_center
            if (
                centre_x - width / 2 < 0
                or centre_y - height / 2 < 0
                or centre_x + width / 2 > self.frame.width_px
                or centre_y + height / 2 > self.frame.height_px
            ):
                raise ValueError("Detection box must remain inside the original frame bounds.")

    def to_dict(self) -> dict[str, Any]:
        """Return the precise JSON-ready cross-team contract."""
        data = asdict(self)
        for detection in data["detections"]:
            detection["bbox_xywh_center"] = list(detection["bbox_xywh_center"])
        return data
