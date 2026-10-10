from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

PointLike = tuple[float, float]


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> PointLike:
        return self.x, self.y


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    bbox: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("detection confidence must be between 0 and 1")


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BoundingBox | tuple[float, float, float, float]
    anchor: Point
    timestamp_s: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bbox, tuple):
            object.__setattr__(self, "bbox", BoundingBox(*self.bbox))
        if self.timestamp_s < 0:
            raise ValueError("track timestamp must be non-negative")


@dataclass(frozen=True)
class Grant:
    grant_id: str
    t_unlock_s: float
    t_relock_s: float

    def __post_init__(self) -> None:
        if not self.grant_id:
            raise ValueError("grant_id must not be empty")
        if self.t_unlock_s < 0 or self.t_relock_s < 0:
            raise ValueError("grant timestamps must be non-negative")
        if self.t_relock_s < self.t_unlock_s:
            raise ValueError("grant relock timestamp must not precede unlock")


@dataclass(frozen=True)
class Crossing:
    track_id: int
    t_cross_s: float
    direction: Literal["in", "out"]

    def __post_init__(self) -> None:
        if self.t_cross_s < 0:
            raise ValueError("crossing timestamp must be non-negative")
        if self.direction not in ("in", "out"):
            raise ValueError("crossing direction must be 'in' or 'out'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    grant_id: str | None
    entries: int
    alert: Literal["none", "tailgating", "unauthorised_entry"]
    crossings: tuple[Crossing, ...] = field(default_factory=tuple)
    t_unlock_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "grant_id": self.grant_id,
            "entries": self.entries,
            "alert": self.alert,
            "crossings": [crossing.to_dict() for crossing in self.crossings],
        }
        if self.t_unlock_s is not None:
            result["t_unlock_s"] = self.t_unlock_s
        return result


@dataclass(frozen=True)
class ThresholdConfig:
    start: PointLike
    end: PointLike
    inside_reference_point: PointLike
    hysteresis_px: float = 12.0

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("threshold endpoints must differ")
        if self.hysteresis_px <= 0:
            raise ValueError("threshold hysteresis must be positive")


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    threshold: ThresholdConfig
    stable_frames: int = 3
    min_displacement_px: float = 25.0
    anchor: str = "bbox_center"
    pre_grace_s: float = 0.0
    post_grace_s: float = 0.5
    detector_model: str = "models/yolox_nano.onnx"
    detector_input_size: int = 416
    detector_confidence: float = 0.3
    detector_nms_iou: float = 0.5
    max_lost_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.stable_frames <= 0 or self.min_displacement_px <= 0:
            raise ValueError("crossing stability settings must be positive")
        if self.anchor not in ("bbox_center", "bottom_center"):
            raise ValueError("anchor must be bbox_center or bottom_center")


def load_camera_config(path: Path) -> CameraConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load camera configuration") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    door = data.get("door", {})
    points = door.get("threshold")
    reference = door.get("inside_reference_point")
    if not isinstance(points, list) or len(points) != 2 or not reference:
        raise ValueError("door threshold and inside_reference_point are required")
    crossing = data.get("crossing", {})
    grant = data.get("grant", {})
    detector = data.get("detector", {})
    tracker = data.get("tracker", {})
    threshold = ThresholdConfig(
        tuple(map(float, points[0])),
        tuple(map(float, points[1])),
        tuple(map(float, reference)),
        float(door.get("hysteresis_px", 12)),
    )
    return CameraConfig(
        camera_id=str(data.get("camera", {}).get("id", path.stem)),
        threshold=threshold,
        stable_frames=int(crossing.get("stable_frames", 3)),
        min_displacement_px=float(crossing.get("min_displacement_px", 25)),
        anchor=str(crossing.get("anchor", "bbox_center")),
        pre_grace_s=float(grant.get("pre_grace_s", 0)),
        post_grace_s=float(grant.get("post_grace_s", 0.5)),
        detector_model=str(detector.get("model_path", "models/yolox_nano.onnx")),
        detector_input_size=int(detector.get("input_size", 416)),
        detector_confidence=float(detector.get("confidence_threshold", 0.3)),
        detector_nms_iou=float(detector.get("nms_iou_threshold", 0.5)),
        max_lost_seconds=float(tracker.get("max_lost_seconds", 1.0)),
    )
