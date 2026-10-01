#!/usr/bin/env python
"""Build the fixed three-image RUOD smoke fixture from COCO training annotations."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SELECTION = (
    ("simple", "012339.jpg", "Single centered fish with one medium-large box"),
    ("multiclass", "012732.jpg", "Five objects across fish, diver, corals, turtle, and jellyfish"),
    ("stress", "008930.jpg", "Ten objects across five classes with small and boundary boxes"),
)

EXPECTED_CLASSES = (
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


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(dataset_root, output_root, overwrite=False):
    annotation_path = dataset_root / "RUOD_ANN" / "instances_train.json"
    image_root = dataset_root / "RUOD_pic" / "train"
    data = json.loads(annotation_path.read_text())
    categories = {item["id"]: item["name"] for item in data["categories"]}
    actual_classes = tuple(categories[index] for index in sorted(categories))
    if actual_classes != EXPECTED_CLASSES:
        raise ValueError("Unexpected RUOD category order: {}".format(actual_classes))

    images_by_name = {item["file_name"]: item for item in data["images"]}
    annotations_by_image = {}
    for item in data["annotations"]:
        if not item.get("iscrowd", 0):
            annotations_by_image.setdefault(item["image_id"], []).append(item)

    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    records = []

    for slot, source_name, reason in SELECTION:
        image = images_by_name[source_name]
        source_path = image_root / source_name
        target_name = "{}_{}".format(slot, source_name)
        target_image = output_images / target_name
        target_label = output_labels / (Path(target_name).stem + ".txt")
        for target in (target_image, target_label):
            if target.exists() and not overwrite:
                raise ValueError("Refusing to overwrite {}".format(target))

        lines = []
        class_names = []
        for annotation in sorted(annotations_by_image[image["id"]], key=lambda item: item["id"]):
            x, y, width, height = annotation["bbox"]
            center_x = (x + width / 2.0) / image["width"]
            center_y = (y + height / 2.0) / image["height"]
            normalized_width = width / image["width"]
            normalized_height = height / image["height"]
            class_id = annotation["category_id"] - 1
            lines.append(
                "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                    class_id, center_x, center_y, normalized_width, normalized_height
                )
            )
            class_names.append(categories[annotation["category_id"]])

        temporary_image = target_image.with_name(target_image.name + ".tmp")
        shutil.copy2(str(source_path), str(temporary_image))
        temporary_image.replace(target_image)
        temporary_label = target_label.with_name(target_label.name + ".tmp")
        temporary_label.write_text("\n".join(lines) + "\n")
        temporary_label.replace(target_label)
        records.append(
            {
                "slot": slot,
                "reason": reason,
                "source_split": "train",
                "source_image_id": image["id"],
                "source_file_name": source_name,
                "output_image": "images/{}".format(target_name),
                "output_label": "labels/{}".format(target_label.name),
                "width": image["width"],
                "height": image["height"],
                "box_count": len(lines),
                "classes": sorted(set(class_names)),
                "image_sha256": checksum(target_image),
                "label_sha256": checksum(target_label),
            }
        )

    manifest = {
        "dataset": "RUOD",
        "source_annotations": str(annotation_path),
        "class_order": list(EXPECTED_CLASSES),
        "selection": records,
    }
    manifest_path = output_root / "selection_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise ValueError("Refusing to overwrite {}".format(manifest_path))
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.dataset_root, args.output_root, args.overwrite)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    for item in manifest["selection"]:
        print("{slot}: {source_file_name} -> {box_count} boxes, {classes}".format(**item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
