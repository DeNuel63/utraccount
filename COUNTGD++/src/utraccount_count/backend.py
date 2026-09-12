"""Protocol implemented by the selected official CountGD++ integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class RawDetection:
    """Normalised candidate produced before CountGD++ thresholding."""

    bbox_xywh_center: tuple[float, float, float, float]
    label: str
    confidence: float
    negative_confidence: float = 0.0


@dataclass(frozen=True)
class RawInference:
    candidates: Sequence[RawDetection]
    model_response: dict[str, Any]


class CountGDBackend(Protocol):
    """Minimal adapter surface; keep repository-specific code behind this protocol."""

    def infer(self, image_path: Path, *, prompt: str) -> RawInference:
        """Run one original-resolution frame and return unfiltered candidates."""
