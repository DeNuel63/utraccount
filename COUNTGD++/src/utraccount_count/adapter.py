"""Convert repository-specific CountGD++ results into the shared detection contract."""

from __future__ import annotations

import json
from pathlib import Path

from .backend import CountGDBackend
from .contracts import Detection, DetectionBatch, FrameInfo, InferenceSource


class CountGDAdapter:
    def __init__(self, backend: CountGDBackend, *, model_version: str) -> None:
        self._backend = backend
        self._model_version = model_version

    def infer_frame(
        self,
        image_path: Path,
        *,
        sequence_id: str,
        frame_index: int,
        frame: FrameInfo,
        prompt: str,
        confidence_threshold: float,
        raw_output_path: Path,
    ) -> DetectionBatch:
        """Persist raw model output, then threshold candidates into one frame batch."""
        raw = self._backend.infer(image_path, prompt=prompt)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(json.dumps(raw.model_response, indent=2), encoding="utf-8")

        detections: list[Detection] = []
        for ordinal, candidate in enumerate(raw.candidates):
            if (
                candidate.confidence < confidence_threshold
                or candidate.confidence <= candidate.negative_confidence
            ):
                continue
            clipped_box = self._clip_to_frame(candidate.bbox_xywh_center, frame)
            if clipped_box is None:
                continue
            detections.append(
                Detection(
                    detection_id=f"{sequence_id}-{frame_index:06d}-{ordinal:03d}",
                    bbox_xywh_center=clipped_box,
                    label=candidate.label,
                    confidence=candidate.confidence,
                )
            )
        return DetectionBatch(
            sequence_id=sequence_id,
            frame_index=frame_index,
            frame=frame,
            source=InferenceSource(
                model_version=self._model_version,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                raw_output_uri=str(raw_output_path),
            ),
            detections=tuple(detections),
        )

    @staticmethod
    def _clip_to_frame(
        bbox_xywh_center: tuple[float, float, float, float], frame: FrameInfo
    ) -> tuple[float, float, float, float] | None:
        """Clip accepted boxes to the original frame; discard empty intersections."""
        centre_x, centre_y, width, height = bbox_xywh_center
        left = max(0.0, centre_x - width / 2)
        top = max(0.0, centre_y - height / 2)
        right = min(float(frame.width_px), centre_x + width / 2)
        bottom = min(float(frame.height_px), centre_y + height / 2)
        if right <= left or bottom <= top:
            return None
        return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
