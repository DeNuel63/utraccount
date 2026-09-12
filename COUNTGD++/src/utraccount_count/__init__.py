"""CountGD++ integration primitives for U-TracCount."""

from .adapter import CountGDAdapter
from .contracts import Detection, DetectionBatch, FrameInfo, InferenceSource
from .covtrack_boundary import (
    COVTrackCompatibilityError,
    COVTrackDetections,
    to_covtrack_detections,
    validate_covtrack_sequence,
)
from .countgdplusplus import CountGDPlusPlusBackend, write_detection_batch
from .dataset_registry import DatasetRegister, load_dataset_register
from .evaluation import CountMetrics, LocalisationMetrics, evaluate_counts, evaluate_localisation, intersection_over_union
from .run_manifest import RunManifest
from .validation import ContractValidationError, load_detection_batch

__all__ = [
    "CountGDAdapter",
    "CountGDPlusPlusBackend",
    "COVTrackCompatibilityError",
    "COVTrackDetections",
    "ContractValidationError",
    "CountMetrics",
    "DatasetRegister",
    "Detection",
    "DetectionBatch",
    "FrameInfo",
    "InferenceSource",
    "LocalisationMetrics",
    "RunManifest",
    "evaluate_counts",
    "evaluate_localisation",
    "intersection_over_union",
    "load_detection_batch",
    "load_dataset_register",
    "to_covtrack_detections",
    "validate_covtrack_sequence",
    "write_detection_batch",
]
