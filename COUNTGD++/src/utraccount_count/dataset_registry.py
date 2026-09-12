"""Dataset registration and split-isolation checks for reproducible experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRegister:
    """Records the provenance and non-overlapping split membership of a dataset."""

    dataset_name: str
    dataset_version: str
    licence: str
    intended_use: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    label_types: tuple[str, ...]
    preprocessing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.dataset_name, self.dataset_version, self.licence, self.intended_use)):
            raise ValueError("Dataset name, version, licence, and intended use are required.")
        if not self.label_types:
            raise ValueError("At least one label type is required.")
        splits = {"train": set(self.train_ids), "validation": set(self.validation_ids), "test": set(self.test_ids)}
        if any(not values for values in splits.values()):
            raise ValueError("Train, validation, and test splits must each contain at least one ID.")
        if sum(len(values) for values in splits.values()) != len(set().union(*splits.values())):
            raise ValueError("Dataset IDs must not appear in more than one split.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("train_ids", "validation_ids", "test_ids", "label_types", "preprocessing"):
            data[key] = list(data[key])
        return data

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_dataset_register(path: Path) -> DatasetRegister:
    """Read a dataset register and enforce its split-isolation invariants."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dataset register root must be an object.")
        expected = {
            "dataset_name", "dataset_version", "licence", "intended_use", "train_ids",
            "validation_ids", "test_ids", "label_types", "preprocessing",
        }
        if set(payload) != expected:
            raise ValueError("Dataset register fields do not match the v1 project format.")
        return DatasetRegister(
            **{key: tuple(value) if isinstance(value, list) else value for key, value in payload.items()}
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid dataset register: {path}") from error
