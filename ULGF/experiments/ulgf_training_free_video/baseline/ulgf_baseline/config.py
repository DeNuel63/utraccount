from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


DEFAULT_RUOD_CLASSES: Tuple[str, ...] = (
    "holothurian",
    "echinus",
    "scallop",
    "starfish",
    "fish",
    "corals",
    "diver",
    "cuttlefish",
    "turtle",
    "jellyfish",
)


@dataclass(frozen=True)
class BaselineConfig:
    checkpoint: Path
    image_dir: Path
    label_dir: Path
    output_dir: Path
    classes: Sequence[str] = DEFAULT_RUOD_CLASSES
    device: str = "cuda"
    seed: int = 0
    seed_file: Optional[Path] = None
    samples_per_layout: int = 1
    guidance_scale: Optional[float] = None
    inference_steps: Optional[int] = None
    overwrite: bool = False
    disable_safety_checker: bool = False

    @property
    def output_image_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def output_label_dir(self) -> Path:
        return self.output_dir / "labels"
