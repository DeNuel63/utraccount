"""Strict reader for detection-batch files at the CountGD++/COVTrack boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Detection, DetectionBatch, FrameInfo, InferenceSource, SCHEMA_VERSION


class ContractValidationError(ValueError):
    """Raised when a JSON file cannot be used safely by the tracking boundary."""


def load_detection_batch(path: Path) -> DetectionBatch:
    """Read a v1 JSON batch and validate its structural and semantic contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"Cannot read detection batch: {path}") from error
    if not isinstance(payload, dict):
        raise ContractValidationError("Detection batch root must be a JSON object.")
    _require_exact_keys(payload, {"schema_version", "sequence_id", "frame_index", "frame", "source", "detections"}, "batch")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractValidationError(f"Unsupported schema version: {payload['schema_version']!r}")
    if not isinstance(payload["frame"], dict) or not isinstance(payload["source"], dict):
        raise ContractValidationError("Frame and source must be JSON objects.")
    _require_exact_keys(payload["frame"], {"uri", "width_px", "height_px", "timestamp_ms"}, "frame")
    _require_exact_keys(
        payload["source"],
        {"model_name", "model_version", "prompt", "confidence_threshold", "box_format", "coordinate_space", "raw_output_uri"},
        "source",
    )
    if not isinstance(payload["detections"], list):
        raise ContractValidationError("Detections must be a JSON array.")
    try:
        frame = FrameInfo(**payload["frame"])
        source = InferenceSource(**payload["source"])
        detections = tuple(_parse_detection(item) for item in payload["detections"])
        return DetectionBatch(
            sequence_id=payload["sequence_id"],
            frame_index=payload["frame_index"],
            frame=frame,
            source=source,
            detections=detections,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractValidationError("Detection batch violates the v1 contract.") from error


def _parse_detection(payload: Any) -> Detection:
    if not isinstance(payload, dict):
        raise ContractValidationError("Each detection must be a JSON object.")
    _require_exact_keys(
        payload,
        {"detection_id", "bbox_xywh_center", "label", "confidence", "ground_truth_instance_id"},
        "detection",
    )
    box = payload["bbox_xywh_center"]
    if not isinstance(box, list) or len(box) != 4:
        raise ContractValidationError("bbox_xywh_center must contain exactly four values.")
    return Detection(
        detection_id=payload["detection_id"],
        bbox_xywh_center=tuple(float(value) for value in box),
        label=payload["label"],
        confidence=float(payload["confidence"]),
        ground_truth_instance_id=payload["ground_truth_instance_id"],
    )


def _require_exact_keys(payload: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ContractValidationError(f"Invalid {location} fields; missing={sorted(missing)}, extra={sorted(extra)}")
