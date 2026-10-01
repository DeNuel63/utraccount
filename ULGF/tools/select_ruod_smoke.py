#!/usr/bin/env python
"""Rank RUOD training images for the three-slot ULGF smoke fixture."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def analyze(dataset_root):
    annotation_path = dataset_root / "RUOD_ANN" / "instances_train.json"
    image_root = dataset_root / "RUOD_pic" / "train"
    data = json.loads(annotation_path.read_text())
    categories = {item["id"]: item["name"] for item in data["categories"]}
    annotations = collections.defaultdict(list)
    for annotation in data["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[annotation["image_id"]].append(annotation)

    candidates = collections.defaultdict(list)
    for image in data["images"]:
        image_annotations = annotations[image["id"]]
        count = len(image_annotations)
        width, height = image["width"], image["height"]
        names = {categories[item["category_id"]] for item in image_annotations}
        areas = [item["bbox"][2] * item["bbox"][3] / (width * height) for item in image_annotations]
        boundary = any(
            item["bbox"][0] / width < 0.03
            or item["bbox"][1] / height < 0.03
            or (item["bbox"][0] + item["bbox"][2]) / width > 0.97
            or (item["bbox"][1] + item["bbox"][3]) / height > 0.97
            for item in image_annotations
        )
        central = all(
            item["bbox"][0] / width > 0.05
            and item["bbox"][1] / height > 0.05
            and (item["bbox"][0] + item["bbox"][2]) / width < 0.95
            and (item["bbox"][1] + item["bbox"][3]) / height < 0.95
            for item in image_annotations
        )
        record = {
            "image_id": image["id"],
            "file_name": image["file_name"],
            "path": str(image_root / image["file_name"]),
            "width": width,
            "height": height,
            "box_count": count,
            "classes": sorted(names),
            "relative_areas": [round(value, 6) for value in areas],
            "boundary_box": boundary,
        }
        if 1 <= count <= 2 and "fish" in names and central and min(areas) > 0.005:
            candidates["simple"].append((sum(abs(value - 0.08) for value in areas), record))
        if 3 <= count <= 6 and len(names) >= 2 and min(areas) > 0.001:
            candidates["multi"].append((-len(names) + sum(areas), record))
        if 8 <= count <= 12 and len(names) >= 2 and min(areas) < 0.01 and boundary:
            candidates["stress"].append((abs(count - 10) - len(names), record))

    counts = [len(annotations[item["id"]]) for item in data["images"]]
    summary = {
        "image_count": len(data["images"]),
        "annotation_count": len(data["annotations"]),
        "categories": categories,
        "empty_images": sum(count == 0 for count in counts),
        "box_count_min": min(counts),
        "box_count_median": statistics.median(counts),
        "box_count_max": max(counts),
        "existing_images": sum((image_root / item["file_name"]).is_file() for item in data["images"]),
    }
    ranked = {
        kind: [record for _, record in sorted(rows, key=lambda item: item[0])]
        for kind, rows in candidates.items()
    }
    return summary, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    summary, ranked = analyze(args.dataset_root)
    print(json.dumps({"summary": summary, "candidates": {key: value[: args.limit] for key, value in ranked.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
