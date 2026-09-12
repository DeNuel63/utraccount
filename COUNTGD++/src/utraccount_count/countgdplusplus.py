"""Concrete CountGD++ text-prompt backend for the U-TracCount baseline.

The official repository is intentionally imported lazily: its CUDA/PyTorch
dependencies are required only in the runtime that performs inference, not for
local contract tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .backend import RawDetection, RawInference


@contextmanager
def _repository_context(repository_root: Path) -> Iterator[None]:
    """Make CountGD++'s relative imports/configuration work without copying it."""
    previous_directory = Path.cwd()
    repository_string = str(repository_root)
    sys.path.insert(0, repository_string)
    try:
        os.chdir(repository_root)
        yield
    finally:
        os.chdir(previous_directory)
        sys.path.remove(repository_string)


class CountGDPlusPlusBackend:
    """Run official CountGD++ with a positive text prompt and no exemplars.

    This is the first, controlled baseline.  The model exposes all query boxes
    and scores; this backend records all of them so thresholds can be revisited
    later without rerunning inference.
    """

    def __init__(
        self,
        repository_root: Path,
        checkpoint_path: Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = device
        self._model: Any | None = None
        self._transform: Any | None = None
        self._app: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.repository_root / "app.py").is_file():
            raise FileNotFoundError(f"CountGD++ app.py not found: {self.repository_root}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"CountGD++ checkpoint not found: {self.checkpoint_path}")

        with _repository_context(self.repository_root):
            app = importlib.import_module("app")
            arguments = app.get_args_parser().parse_args([])
            arguments.pretrain_model_path = str(self.checkpoint_path)
            arguments.device = self.device
            model, transform = app.build_model_and_transforms(arguments)
            self._model = model.to(self.device)
            self._transform = transform
            self._app = app

    @staticmethod
    def _decode_raw_output(
        output: Any,
        *,
        prompt: str,
        image_width: int,
        image_height: int,
    ) -> RawInference:
        """Convert CountGD++ normalized query output into raw pixel candidates."""
        logits = output["pred_logits"].sigmoid()[0]
        boxes = output["pred_boxes"][0]
        token_ids = output["input_ids"][0]

        separator_indices = (token_ids == 1012).nonzero(as_tuple=True)[0]
        if len(separator_indices) == 0:
            raise ValueError("CountGD++ output lacks the expected prompt separator token.")
        separator_index = int(separator_indices[0])
        positive_scores = logits[:, : separator_index + 1].max(dim=-1).values
        negative_logits = logits[:, separator_index + 1 :]
        negative_scores = (
            negative_logits.max(dim=-1).values
            if negative_logits.shape[-1] > 0
            else positive_scores.new_zeros(positive_scores.shape)
        )

        candidates: list[RawDetection] = []
        raw_queries: list[dict[str, Any]] = []
        for index, (box, positive, negative) in enumerate(zip(boxes, positive_scores, negative_scores)):
            centre_x, centre_y, width, height = box.detach().cpu().tolist()
            candidate = RawDetection(
                bbox_xywh_center=(
                    float(centre_x * image_width),
                    float(centre_y * image_height),
                    float(width * image_width),
                    float(height * image_height),
                ),
                label=prompt,
                confidence=float(positive.detach().cpu().item()),
                negative_confidence=float(negative.detach().cpu().item()),
            )
            candidates.append(candidate)
            raw_queries.append(
                {
                    "query_index": index,
                    "bbox_xywh_center_normalized": [centre_x, centre_y, width, height],
                    "positive_confidence": candidate.confidence,
                    "negative_confidence": candidate.negative_confidence,
                }
            )

        return RawInference(
            candidates=candidates,
            model_response={
                "format": "utraccount.countgdplusplus.raw-queries/v1",
                "prompt": {"positive_text": prompt, "negative_texts": []},
                "query_count": len(raw_queries),
                "queries": raw_queries,
            },
        )

    def infer(self, image_path: Path, *, prompt: str) -> RawInference:
        self._load()
        assert self._app is not None and self._model is not None and self._transform is not None
        if not prompt.strip():
            raise ValueError("A non-empty positive text prompt is required.")

        with _repository_context(self.repository_root):
            torch = importlib.import_module("torch")
            pil_image = importlib.import_module("PIL.Image").open(image_path).convert("RGB")
            input_image, input_exemplar_image, positive_exemplars = self._app.preprocess(
                self._transform, pil_image, {"image": pil_image, "points": []}
            )
            nested_tensor_from_tensor_list = importlib.import_module("util.misc").nested_tensor_from_tensor_list
            with torch.no_grad():
                output = self._model(
                    nested_tensor_from_tensor_list(input_image.unsqueeze(0).to(self.device)),
                    nested_tensor_from_tensor_list(input_exemplar_image.unsqueeze(0).to(self.device)),
                    [positive_exemplars.to(self.device)],
                    [],
                    [],
                    captions=[prompt.strip() + " . "],
                )
        return self._decode_raw_output(
            output,
            prompt=prompt.strip(),
            image_width=pil_image.width,
            image_height=pil_image.height,
        )


def write_detection_batch(batch: Any, output_path: Path) -> None:
    """Persist the public tracker contract after raw output was saved."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(batch.to_dict(), indent=2), encoding="utf-8")
