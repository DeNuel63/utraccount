from pathlib import Path
from typing import List, Sequence, Tuple


LayoutObject = Tuple[str, float, float, float, float]


def read_yolo_layout(label_path: Path, classes: Sequence[str], max_objects: int = 12) -> List[LayoutObject]:
    objects: List[LayoutObject] = []
    for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("{}:{} must contain five YOLO fields".format(label_path, line_number))
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(classes):
            raise ValueError("{}:{} has unknown class id {}".format(label_path, line_number, class_id))
        center_x, center_y, width, height = (float(value) for value in fields[1:])
        if width <= 0 or height <= 0:
            raise ValueError("{}:{} has a non-positive box size".format(label_path, line_number))
        x1 = max(0.0, center_x - width / 2.0)
        y1 = max(0.0, center_y - height / 2.0)
        x2 = min(1.0, center_x + width / 2.0)
        y2 = min(1.0, center_y + height / 2.0)
        if not (x1 < x2 and y1 < y2):
            raise ValueError("{}:{} produces an invalid clipped box".format(label_path, line_number))
        if (x2 - x1) * (y2 - y1) < 0.0001:
            continue
        objects.append((classes[class_id], x1, y1, x2, y2))
        if len(objects) >= max_objects:
            break
    if not objects:
        raise ValueError("No usable objects found in {}".format(label_path))
    return objects


def to_yolo_lines(objects: Sequence[LayoutObject], classes: Sequence[str]) -> List[str]:
    class_ids = {name: index for index, name in enumerate(classes)}
    lines = []
    for class_name, x1, y1, x2, y2 in objects:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        lines.append(
            "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                class_ids[class_name], center_x, center_y, width, height
            )
        )
    return lines
