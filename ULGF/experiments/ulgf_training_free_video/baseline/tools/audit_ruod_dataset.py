"""Audit RUOD and optionally export EXIF-free stored-pixel images.

Python >=3.7, Pillow only. Originals are never modified. Re-running resumes
exports by checking existing pixels. JPEGs are not recompressed.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageOps


SPLITS = {
    "train": ("RUOD_ANN/instances_train.json", "RUOD_pic/train"),
    "test": ("RUOD_ANN/instances_test.json", "RUOD_pic/test"),
    "environment_light": ("Environmet_ANN/instances_light.json", "Environment_pic/light"),
    "environment_color": ("Environmet_ANN/instances_color.json", "Environment_pic/color"),
    "environment_blur": ("Environmet_ANN/instances_blur.json", "Environment_pic/blur"),
}


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256((str(rgb.size) + "RGB").encode() + rgb.tobytes()).hexdigest()


def export_filename(image_id, source_mode, source_format):
    """Single filename policy shared by export and reconstructed split contracts."""
    suffix = '.jpg' if source_format == 'JPEG' and source_mode == 'RGB' else '.png'
    return str(image_id) + suffix


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_jpeg_exif(blob):
    """Remove EXIF/XMP APP1 metadata; preserve JPEG image data and ICC profile.

    Pillow can reconstruct EXIF orientation from XMP after EXIF is removed.
    Remove standard and extended XMP so decoders agree on stored orientation.
    """
    if blob[:2] != b'\xff\xd8':
        raise ValueError('Not a JPEG')
    result, offset = bytearray(blob[:2]), 2
    while offset < len(blob):
        start = offset
        if blob[offset] != 255:
            raise ValueError('Invalid JPEG marker')
        while offset < len(blob) and blob[offset] == 255:
            offset += 1
        if offset >= len(blob):
            raise ValueError('Truncated JPEG marker')
        marker = blob[offset]
        offset += 1
        if marker in (0xDA, 0xD9):  # Preserve scan data verbatim.
            result.extend(blob[start:])
            return bytes(result)
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            result.extend(blob[start:offset])
            continue
        length = int.from_bytes(blob[offset:offset + 2], 'big')
        end = offset + length
        if length < 2 or end > len(blob):
            raise ValueError('Invalid JPEG segment length')
        payload = blob[offset + 2:end]
        orientation_metadata = marker == 0xE1 and payload.startswith((
            b'Exif\x00\x00',
            b'http://ns.adobe.com/xap/1.0/\x00',
            b'http://ns.adobe.com/xmp/extension/\x00',
        ))
        if not orientation_metadata:
            result.extend(blob[start:end])
        offset = end
    raise ValueError('JPEG has no scan/end marker')


def safe_path(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def valid_box(box, size):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return False
    x, y, w, h = box
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= size[0] and y + h <= size[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def decode_record(arguments):
    root, item = arguments
    try:
        path = safe_path(root, item["file_name"])
        metadata = {"file_sha256": file_hash(path)}
        with Image.open(path) as image:
            image.load()
            orientation = image.getexif().get(274, 1)
            rgb = image.convert("RGB")
            metadata.update(stored_size=list(image.size), exif_orientation=orientation,
                            source_format=image.format,
                            source_mode=image.mode, stored_pixels_sha256=pixel_hash(rgb))
            metadata["display_pixels_sha256"] = (metadata["stored_pixels_sha256"]
                if orientation == 1 else pixel_hash(ImageOps.exif_transpose(image)))
        return path, metadata, rgb
    except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as error:
        return error


def decoded_records(root, items):
    # Bounded batches: never retain decoded rasters for the entire dataset.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(items), 4):
            batch = items[start:start + 4]
            results = executor.map(decode_record, [(root, item) for item in batch])
            for item, decoded in zip(batch, results):
                yield item, decoded


def audit(root, output, derive=False, approvals=None):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    approvals = approvals or {}
    output.mkdir(parents=True, exist_ok=True)
    policy = {"source": str(root), "convention": "stored RGB pixels, unchanged boxes, EXIF-free JPEG without recompression; PNG fallback",
              "approved_raw_coordinates": approvals, "derive": derive}
    policy_path = output / "policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Output belongs to a different policy; use a new output directory")
    write_json(policy_path, policy)
    groups = {kind: defaultdict(list) for kind in ("file", "stored_pixels", "display_pixels")}
    summaries = {}
    total_quarantined = 0
    for split, (annotation_name, image_folder) in SPLITS.items():
        annotation_path = root / annotation_name
        if not annotation_path.is_file():
            summaries[split] = {"error": "annotation file missing"}
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        for key in ("images", "annotations", "categories"):
            ids = [item["id"] for item in data[key]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs: {} {}".format(split, key))
        images_by_id = {item["id"]: item for item in data["images"]}
        categories = {item["id"] for item in data["categories"]}
        by_image = defaultdict(list)
        for ann in data["annotations"]:
            if ann["image_id"] not in images_by_id or ann["category_id"] not in categories:
                raise ValueError("Orphan annotation in {}: {}".format(split, ann["id"]))
            by_image[ann["image_id"]].append(ann)
        records, kept_images, kept_annotations = [], [], []
        orientations, problems = Counter(), Counter()
        expected_paths = set()
        for index, (item, decoded) in enumerate(decoded_records(root / image_folder, data["images"])):
            record = {"id": item["id"], "file_name": item["file_name"],
                      "declared_size": [item["width"], item["height"]], "issues": []}
            anns = by_image[item["id"]]
            rgb = None
            try:
                path = safe_path(root / image_folder, item["file_name"])
                expected_paths.add(path)
                if isinstance(decoded, Exception):
                    raise decoded
                path, metadata, rgb = decoded
                record.update(metadata)
                orientation = record["exif_orientation"]
                orientations[str(orientation)] += 1
                identity = {"split": split, "id": item["id"], "file_name": item["file_name"]}
                for kind in groups:
                    groups[kind][record[kind + "_sha256"]].append(identity)
                if rgb.size != (item["width"], item["height"]):
                    record["dimension_mismatch"] = True
                approved = approvals.get(split + ":" + str(item["id"]))
                # Approval is bound to exact source bytes, not merely a filename.
                raw_approved = approved == record["file_sha256"]
                record["raw_coordinate_approval"] = raw_approved
                if orientation not in range(1, 9) and not (orientation == 0 and raw_approved):
                    record["issues"].append("invalid_exif_orientation")
                if (orientation != 1 or record.get("dimension_mismatch")) and not raw_approved:
                    record["issues"].append("coordinate_convention_needs_review")
                bad = [a["id"] for a in anns if not valid_box(a["bbox"], rgb.size)]
                if bad:
                    record["issues"].append("invalid_stored_coordinate_boxes")
                    record["invalid_annotation_ids"] = bad
                if any(a.get("segmentation") or a.get("keypoints") for a in anns):
                    record["issues"].append("non_bbox_geometry_needs_review")
            except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as error:
                record["issues"].append("read_or_schema_error")
                record["error"] = str(error)
            if record["issues"]:
                total_quarantined += 1
                problems.update(record["issues"])
                record["status"] = "QUARANTINED"
            else:
                record["status"] = "ACCEPTED"
                use_jpeg = record.get("source_format") == "JPEG" and record.get("source_mode") == "RGB"
                filename = export_filename(item["id"], record.get("source_mode"), record.get("source_format"))
                if derive:
                    record["derived_file"] = "images/{}/{}".format(split, filename)
                converted = dict(item, file_name=filename, width=rgb.width, height=rgb.height)
                kept_images.append(converted)
                kept_annotations.extend(anns)
                if derive:
                    destination = output / "images" / split / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        with Image.open(destination) as existing:
                            if pixel_hash(existing) != record["stored_pixels_sha256"] or existing.getexif():
                                raise ValueError("Existing export differs: " + str(destination))
                    else:
                        if shutil.disk_usage(output).free < rgb.width * rgb.height * 6 + 256 * 1024 ** 2:
                            raise OSError("Insufficient free space for safe image export; rerun on a larger disk")
                        temporary = destination.with_suffix(".tmp")
                        if use_jpeg:
                            temporary.write_bytes(strip_jpeg_exif(path.read_bytes()))
                        else:
                            clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
                            clean.save(temporary, format="PNG", compress_level=1)
                        with Image.open(temporary) as check:
                            if pixel_hash(check) != record["stored_pixels_sha256"] or check.getexif():
                                raise ValueError(
                                    "Export validation failed: source={}, temporary={}, "
                                    "pixels_identical={}, remaining_exif={}".format(
                                        path, temporary,
                                        pixel_hash(check) == record["stored_pixels_sha256"],
                                        dict(check.getexif())))
                        temporary.replace(destination)
            records.append(record)
            if (index + 1) % 250 == 0:
                print("{}: {}/{}".format(split, index + 1, len(data["images"])), flush=True)
                write_json(output / "audit" / (split + ".json"), records)
        present = {p.resolve() for p in (root / image_folder).rglob("*") if p.is_file()}
        summary = {"declared_images": len(data["images"]), "accepted_images": len(kept_images),
                   "accepted_annotations": len(kept_annotations),
                   "quarantined_annotations": len(data["annotations"]) - len(kept_annotations),
                   "quarantined_images": len(records) - len(kept_images),
                   "orientations": dict(orientations), "issues": dict(problems),
                   "dimension_mismatches": sum(bool(r.get("dimension_mismatch")) for r in records),
                   "unreferenced_files": sorted(str(p.relative_to(root)) for p in present - expected_paths),
                   "annotation_sha256": file_hash(annotation_path)}
        summaries[split] = summary
        write_json(output / "audit" / (split + ".json"), records)
        if derive:
            derived = dict(data, images=kept_images, annotations=kept_annotations)
            write_json(output / "annotations" / ("instances_" + split + ".json"), derived)
    duplicates = {kind: [v for v in hashes.values() if len(v) > 1] for kind, hashes in groups.items()}
    leakage = [g for g in duplicates["display_pixels"]
               if {"train", "test"}.issubset({x["split"] for x in g})]
    report = {"splits": summaries, "quarantined_images": total_quarantined,
              "duplicate_groups": {k: len(v) for k, v in duplicates.items()},
              "train_test_exact_pixel_leakage_groups": len(leakage),
              "near_duplicate_checks": "NOT_PERFORMED: exact hashes do not detect recompressed or cropped copies",
              "training_ready": False,
              "status": "AUDIT_COMPLETE_REVIEW_REQUIRED",
              "note": "Review quarantines, duplicates and category mapping before training. Test exclusions require explicit approval."}
    write_json(output / "duplicates.json", duplicates)
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--approvals", type=Path, help="JSON mapping split:image_id to reviewed source SHA256")
    args = parser.parse_args()
    audit(args.source, args.output, args.derive,
          json.loads(args.approvals.read_text()) if args.approvals else None)
