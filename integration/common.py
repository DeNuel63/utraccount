import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_clip(path, class_map):
    """Read existing ULGF frames; keep requested reference objects separate."""
    from PIL import Image
    path = Path(path).resolve()
    clip = json.loads(path.read_text(encoding='utf-8'))
    if clip.get('schema_version') != '1.0':
        raise ValueError('Expected ULGF clip schema 1.0')
    if not clip.get('clip_id') or not clip.get('frames'):
        raise ValueError('A named, non-empty clip is required')
    fps = clip['fps']
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    records = []
    for index, frame in enumerate(clip['frames']):
        if type(frame['frame_index']) is not int or frame['frame_index'] != index:
            raise ValueError('Frames must be consecutive and ordered from zero')
        image = (path.parent / frame['image_path']).resolve()
        image.relative_to(path.parent)
        with Image.open(str(image)) as pixels:
            if pixels.size != (clip['width'], clip['height']):
                raise ValueError('Frame dimensions differ from manifest: ' + str(image))
        references = []
        ids = set()
        for obj in frame['objects']:
            if obj['class_name'] not in class_map or obj['class_id'] != class_map[obj['class_name']]:
                raise ValueError('ULGF class IDs must match the shared class mapping')
            if obj['instance_id'] in ids:
                raise ValueError('Duplicate reference instance ID')
            ids.add(obj['instance_id'])
            x1, y1, x2, y2 = obj['bbox_xyxy']
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError('Reference boxes must be normalized xyxy')
            references.append(dict(obj, bbox_xyxy_pixel=[x1 * clip['width'], y1 * clip['height'], x2 * clip['width'], y2 * clip['height']]))
        records.append(dict(frame_index=index, image=str(image), timestamp_ms=round(index * 1000 / fps), references=references))
    return clip, records


def validate_class_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('A non-empty class mapping is required')
    if any(not isinstance(k, str) or not k.strip() or type(v) is not int or v < 0 for k, v in value.items()):
        raise ValueError('Class mapping requires text names and non-negative integer IDs')
    if len(set(value.values())) != len(value):
        raise ValueError('Class IDs must be unique')
    return value
