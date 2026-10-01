import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectInstance:
    instance_id: int
    class_id: int
    class_name: str
    bbox_xyxy: BBox
    visibility: float = 1.0

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ObjectInstance":
        return cls(
            instance_id=int(value["instance_id"]),
            class_id=int(value["class_id"]),
            class_name=str(value["class_name"]),
            bbox_xyxy=tuple(float(item) for item in value["bbox_xyxy"]),
            visibility=float(value.get("visibility", 1.0)),
        )


@dataclass(frozen=True)
class FrameLayout:
    frame_index: int
    image_path: str
    objects: Tuple[ObjectInstance, ...]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FrameLayout":
        return cls(
            frame_index=int(value["frame_index"]),
            image_path=str(value["image_path"]),
            objects=tuple(ObjectInstance.from_dict(item) for item in value["objects"]),
        )


@dataclass(frozen=True)
class MediumSpec:
    description: str

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MediumSpec":
        return cls(description=str(value["description"]))


@dataclass(frozen=True)
class GenerationMetadata:
    checkpoint: str
    config: str
    temporal_mode: str = "layout_only"
    source_image: Optional[str] = None
    motion: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "GenerationMetadata":
        return cls(
            checkpoint=str(value.get("checkpoint", "")),
            config=str(value.get("config", "")),
            temporal_mode=str(value.get("temporal_mode", "layout_only")),
            source_image=value.get("source_image"),
            motion=dict(value.get("motion", {})),
        )


@dataclass(frozen=True)
class ClipManifest:
    clip_id: str
    seed: int
    fps: float
    width: int
    height: int
    medium: MediumSpec
    frames: Tuple[FrameLayout, ...]
    generation: GenerationMetadata
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["seed"] = int(self.seed)
        value["fps"] = float(self.fps)
        value["width"] = int(self.width)
        value["height"] = int(self.height)
        value["frames"] = [
            {
                "frame_index": frame.frame_index,
                "image_path": frame.image_path,
                "objects": [
                    {
                        "instance_id": obj.instance_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "bbox_xyxy": [round(coordinate, 8) for coordinate in obj.bbox_xyxy],
                        "visibility": round(obj.visibility, 8),
                    }
                    for obj in frame.objects
                ],
            }
            for frame in self.frames
        ]
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(self.to_json())
        temporary.replace(path)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ClipManifest":
        return cls(
            schema_version=str(value["schema_version"]),
            clip_id=str(value["clip_id"]),
            seed=int(value["seed"]),
            fps=float(value["fps"]),
            width=int(value["width"]),
            height=int(value["height"]),
            medium=MediumSpec.from_dict(value["medium"]),
            frames=tuple(FrameLayout.from_dict(item) for item in value["frames"]),
            generation=GenerationMetadata.from_dict(value["generation"]),
        )

    @classmethod
    def read(cls, path: Path) -> "ClipManifest":
        return cls.from_dict(json.loads(path.read_text()))


def initial_objects_from_yolo(
    rows: Sequence[Tuple[str, float, float, float, float]], classes: Sequence[str]
) -> Tuple[ObjectInstance, ...]:
    class_ids = {name: index for index, name in enumerate(classes)}
    return tuple(
        ObjectInstance(
            instance_id=index + 1,
            class_id=class_ids[class_name],
            class_name=class_name,
            bbox_xyxy=(x1, y1, x2, y2),
        )
        for index, (class_name, x1, y1, x2, y2) in enumerate(rows)
    )
