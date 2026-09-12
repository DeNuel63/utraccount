"""Reproducible run metadata independent of any specific GPU service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "utraccount.run-manifest/v1"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    model_version: str
    checkpoint_sha256: str
    dataset_name: str
    dataset_version: str
    split: str
    prompt: str
    confidence_threshold: float
    runtime: dict[str, str]
    source_archive_sha256: str | None = None
    random_seed: int | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        required = (self.run_id, self.model_version, self.checkpoint_sha256, self.dataset_name, self.dataset_version, self.split, self.prompt)
        if not all(required):
            raise ValueError("Run manifest identity, checkpoint, dataset, split, and prompt are required.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not self.runtime:
            raise ValueError("Runtime metadata is required.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
