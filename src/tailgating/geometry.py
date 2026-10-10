from __future__ import annotations

from enum import Enum
from math import hypot

from .schemas import PointLike, ThresholdConfig


class Zone(Enum):
    OUTSIDE = "outside"
    BAND = "band"
    INSIDE = "inside"


def signed_distance(point: PointLike, start: PointLike, end: PointLike) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0:
        raise ValueError("threshold endpoints must differ")
    return (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / length


def normalized_distance(point: PointLike, threshold: ThresholdConfig) -> float:
    reference_sign = 1.0 if signed_distance(
        threshold.inside_reference_point, threshold.start, threshold.end
    ) > 0 else -1.0
    # The reference sign keeps INSIDE stable even if endpoints are reversed.
    return signed_distance(point, threshold.start, threshold.end) * reference_sign


def classify_zone(point: PointLike, threshold: ThresholdConfig) -> Zone:
    distance = normalized_distance(point, threshold)
    if distance > threshold.hysteresis_px:
        return Zone.INSIDE
    if distance < -threshold.hysteresis_px:
        return Zone.OUTSIDE
    return Zone.BAND


def crossing_time(
    previous_point: PointLike,
    previous_time: float,
    point: PointLike,
    time: float,
    threshold: ThresholdConfig,
) -> float:
    before = normalized_distance(previous_point, threshold)
    after = normalized_distance(point, threshold)
    denominator = after - before
    if denominator == 0:
        return time
    ratio = max(0.0, min(1.0, -before / denominator))
    return previous_time + ratio * (time - previous_time)


def entry_normal(threshold: ThresholdConfig) -> PointLike:
    dx, dy = threshold.end[0] - threshold.start[0], threshold.end[1] - threshold.start[1]
    length = hypot(dx, dy)
    candidate = (-dy / length, dx / length)
    sign = 1.0 if signed_distance(threshold.inside_reference_point, threshold.start, threshold.end) > 0 else -1.0
    return candidate[0] * sign, candidate[1] * sign
