import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .schema import BBox, ObjectInstance


@dataclass(frozen=True)
class MotionConfig:
    frames: int = 8
    fps: float = 10.0
    max_speed_box_fraction: float = 0.15
    max_acceleration_box_fraction: float = 0.02
    boundary_policy: str = "reflect"

    def validate(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be at least 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_speed_box_fraction < 0 or self.max_acceleration_box_fraction < 0:
            raise ValueError("motion limits must be non-negative")
        if self.boundary_policy not in ("reflect", "clamp"):
            raise ValueError("boundary_policy must be reflect or clamp")


Velocity = Tuple[float, float]


def _initial_velocity(obj: ObjectInstance, rng: random.Random, config: MotionConfig) -> Velocity:
    x1, y1, x2, y2 = obj.bbox_xyxy
    scale = min(x2 - x1, y2 - y1)
    limit = scale * config.max_speed_box_fraction
    return rng.uniform(-limit, limit), rng.uniform(-limit, limit)


def _advance(bbox: BBox, velocity: Velocity, policy: str) -> Tuple[BBox, Velocity]:
    x1, y1, x2, y2 = bbox
    vx, vy = velocity
    nx1, nx2 = x1 + vx, x2 + vx
    ny1, ny2 = y1 + vy, y2 + vy

    if policy == "reflect":
        if nx1 < 0.0 or nx2 > 1.0:
            vx = -vx
            nx1, nx2 = x1 + vx, x2 + vx
        if ny1 < 0.0 or ny2 > 1.0:
            vy = -vy
            ny1, ny2 = y1 + vy, y2 + vy

    width, height = x2 - x1, y2 - y1
    nx1 = min(max(nx1, 0.0), 1.0 - width)
    ny1 = min(max(ny1, 0.0), 1.0 - height)
    nx2, ny2 = nx1 + width, ny1 + height

    if policy == "clamp":
        if nx1 == 0.0 or nx2 == 1.0:
            vx = 0.0
        if ny1 == 0.0 or ny2 == 1.0:
            vy = 0.0
    return (nx1, ny1, nx2, ny2), (vx, vy)


def generate_object_sequences(
    initial_objects: Iterable[ObjectInstance], seed: int, config: MotionConfig
) -> Tuple[Tuple[ObjectInstance, ...], ...]:
    config.validate()
    objects = tuple(initial_objects)
    rng = random.Random(seed)
    velocities: Dict[int, Velocity] = {obj.instance_id: _initial_velocity(obj, rng, config) for obj in objects}
    frames: List[Tuple[ObjectInstance, ...]] = [objects]

    for _frame_index in range(1, config.frames):
        next_objects: List[ObjectInstance] = []
        for obj in frames[-1]:
            x1, y1, x2, y2 = obj.bbox_xyxy
            scale = min(x2 - x1, y2 - y1)
            acceleration_limit = scale * config.max_acceleration_box_fraction
            speed_limit = scale * config.max_speed_box_fraction
            vx, vy = velocities[obj.instance_id]
            vx += rng.uniform(-acceleration_limit, acceleration_limit)
            vy += rng.uniform(-acceleration_limit, acceleration_limit)
            vx = min(max(vx, -speed_limit), speed_limit)
            vy = min(max(vy, -speed_limit), speed_limit)
            bbox, velocity = _advance(obj.bbox_xyxy, (vx, vy), config.boundary_policy)
            velocities[obj.instance_id] = velocity
            next_objects.append(
                ObjectInstance(
                    instance_id=obj.instance_id,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox_xyxy=bbox,
                    visibility=obj.visibility,
                )
            )
        frames.append(tuple(next_objects))
    return tuple(frames)
