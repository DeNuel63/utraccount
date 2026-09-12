"""Detector-only evaluation helpers; no COVTrack dependency is required."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from .contracts import Detection


@dataclass(frozen=True)
class LocalisationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    iou_threshold: float


@dataclass(frozen=True)
class CountMetrics:
    mae: float
    rmse: float
    samples: int


def intersection_over_union(first: Detection, second: Detection) -> float:
    """Calculate IoU for two pixel xywh-centre boxes."""
    first_left, first_top, first_right, first_bottom = _corners(first)
    second_left, second_top, second_right, second_bottom = _corners(second)
    intersection_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_top, second_top))
    intersection = intersection_width * intersection_height
    union = (first_right - first_left) * (first_bottom - first_top)
    union += (second_right - second_left) * (second_bottom - second_top) - intersection
    return intersection / union if union else 0.0


def evaluate_localisation(
    predictions: Iterable[Detection], ground_truth: Iterable[Detection], *, iou_threshold: float = 0.5
) -> LocalisationMetrics:
    """Greedily match same-label boxes by highest IoU for a single frame."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1.")
    predicted = list(predictions)
    truth = list(ground_truth)
    matches = sorted(
        (
            (intersection_over_union(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predicted)
            for target_index, target in enumerate(truth)
            if prediction.label == target.label
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    for overlap, prediction_index, target_index in matches:
        if overlap < iou_threshold:
            break
        if prediction_index not in used_predictions and target_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(target_index)
    true_positives = len(used_predictions)
    false_positives = len(predicted) - true_positives
    false_negatives = len(truth) - true_positives
    precision = true_positives / (true_positives + false_positives) if predicted else 0.0
    recall = true_positives / (true_positives + false_negatives) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return LocalisationMetrics(true_positives, false_positives, false_negatives, precision, recall, f1, iou_threshold)


def evaluate_counts(predicted_counts: Iterable[int], true_counts: Iterable[int]) -> CountMetrics:
    """Return MAE and RMSE for aligned per-frame counts."""
    predictions = list(predicted_counts)
    truth = list(true_counts)
    if not predictions or len(predictions) != len(truth):
        raise ValueError("Predicted and true counts must be non-empty and aligned.")
    errors = [prediction - target for prediction, target in zip(predictions, truth)]
    return CountMetrics(
        mae=sum(abs(error) for error in errors) / len(errors),
        rmse=sqrt(sum(error**2 for error in errors) / len(errors)),
        samples=len(errors),
    )


def _corners(detection: Detection) -> tuple[float, float, float, float]:
    centre_x, centre_y, width, height = detection.bbox_xywh_center
    return (centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2)
