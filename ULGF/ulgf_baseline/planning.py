from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import BaselineConfig


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PreflightError(ValueError):
    """Raised when a baseline run would be invalid or unsafe."""


@dataclass(frozen=True)
class GenerationItem:
    source_image: Path
    source_label: Path
    output_images: Sequence[Path]
    output_label: Path
    seed: int


def _read_seeds(config: BaselineConfig, count: int) -> List[int]:
    if config.seed_file is None:
        return [config.seed + index for index in range(count)]
    if not config.seed_file.is_file():
        raise PreflightError("Seed file does not exist: {}".format(config.seed_file))
    try:
        seeds = [int(line.strip()) for line in config.seed_file.read_text().splitlines() if line.strip()]
    except ValueError as error:
        raise PreflightError("Seed file contains a non-integer value: {}".format(config.seed_file)) from error
    if len(seeds) < count:
        raise PreflightError(
            "Seed file contains {} values but {} images were found".format(len(seeds), count)
        )
    return seeds[:count]


def _output_names(stem: str, suffix: str, samples: int) -> List[str]:
    if samples == 1:
        return ["Geo{}{}".format(stem, suffix)]
    return ["Geo{}_{:02d}{}".format(stem, index, suffix) for index in range(samples)]


def build_generation_plan(config: BaselineConfig) -> List[GenerationItem]:
    if not config.checkpoint.exists():
        raise PreflightError("Checkpoint does not exist: {}".format(config.checkpoint))
    if config.checkpoint.is_dir():
        required_checkpoint_files = ("model_index.json", "generation_config.json")
        missing_checkpoint_files = [
            name for name in required_checkpoint_files if not (config.checkpoint / name).is_file()
        ]
        if missing_checkpoint_files:
            raise PreflightError(
                "Checkpoint directory is incomplete; missing {} in {}".format(
                    ", ".join(missing_checkpoint_files), config.checkpoint
                )
            )
    if not config.image_dir.is_dir():
        raise PreflightError("Image directory does not exist: {}".format(config.image_dir))
    if not config.label_dir.is_dir():
        raise PreflightError("Label directory does not exist: {}".format(config.label_dir))
    if config.samples_per_layout < 1:
        raise PreflightError("samples_per_layout must be at least 1")
    if not config.classes:
        raise PreflightError("At least one class name is required")

    output_root = config.output_dir.resolve()
    source_roots = (config.image_dir.resolve(), config.label_dir.resolve(), config.checkpoint.resolve())
    for source_root in source_roots:
        try:
            output_root.relative_to(source_root)
        except ValueError:
            continue
        raise PreflightError(
            "Output directory must not be the same as or nested inside an input directory: {}".format(source_root)
        )

    images = sorted(
        path
        for path in config.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise PreflightError("No supported images found in {}".format(config.image_dir))

    seeds = _read_seeds(config, len(images))
    items: List[GenerationItem] = []
    missing_labels: List[Path] = []
    conflicts: List[Path] = []
    run_metadata = config.output_dir / "run.json"
    if run_metadata.exists() and not config.overwrite:
        conflicts.append(run_metadata)

    for source_image, seed in zip(images, seeds):
        source_label = config.label_dir / (source_image.stem + ".txt")
        if not source_label.is_file():
            missing_labels.append(source_label)
            continue
        output_images = [
            config.output_image_dir / name
            for name in _output_names(source_image.stem, source_image.suffix.lower(), config.samples_per_layout)
        ]
        output_label = config.output_label_dir / ("Geo" + source_image.stem + ".txt")
        if not config.overwrite:
            conflicts.extend(path for path in list(output_images) + [output_label] if path.exists())
        items.append(GenerationItem(source_image, source_label, output_images, output_label, seed))

    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:5])
        raise PreflightError("Missing labels for {} image(s): {}".format(len(missing_labels), preview))
    if conflicts:
        preview = ", ".join(str(path) for path in conflicts[:5])
        raise PreflightError(
            "Refusing to overwrite {} existing output(s): {}. Pass --overwrite explicitly to replace them.".format(
                len(conflicts), preview
            )
        )
    return items


def describe_plan(config: BaselineConfig, items: Iterable[GenerationItem]) -> str:
    items = list(items)
    return "\n".join(
        [
            "Baseline generation plan",
            "  checkpoint: {}".format(config.checkpoint),
            "  device: {}".format(config.device),
            "  inputs: {}".format(len(items)),
            "  samples per layout: {}".format(config.samples_per_layout),
            "  output: {}".format(config.output_dir),
            "  overwrite: {}".format(config.overwrite),
            "  safety checker disabled: {}".format(config.disable_safety_checker),
        ]
    )
