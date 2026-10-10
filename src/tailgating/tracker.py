from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from .schemas import BoundingBox, Detection, Point, Track


@dataclass
class _State:
    track_id: int
    bbox: BoundingBox
    center: Point
    previous_center: Point
    confidence: float
    last_frame: int
    missed: int = 0


class CentroidTracker:
    def __init__(self, max_lost_frames: int = 30, max_center_distance: float = 100.0) -> None:
        self.max_lost_frames = max_lost_frames
        self.max_center_distance = max_center_distance
        self._next_id = 1
        self._states: dict[int, _State] = {}

    @staticmethod
    def _iou(left: BoundingBox, right: BoundingBox) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return intersection / (left.area + right.area - intersection) if intersection else 0.0

    def update(self, detections: Sequence[Detection], frame_index: int, timestamp_s: float) -> list[Track]:
        unmatched_tracks = set(self._states)
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []
        for track_id, state in self._states.items():
            velocity_x = state.center.x - state.previous_center.x
            velocity_y = state.center.y - state.previous_center.y
            predicted = Point(state.center.x + velocity_x, state.center.y + velocity_y)
            for detection_index, detection in enumerate(detections):
                center = detection.bbox.center
                distance = hypot(center.x - predicted.x, center.y - predicted.y)
                iou = self._iou(state.bbox, detection.bbox)
                if distance <= self.max_center_distance or iou >= 0.1:
                    candidates.append((distance - iou * self.max_center_distance, track_id, detection_index))

        for _, track_id, detection_index in sorted(candidates):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            detection = detections[detection_index]
            state = self._states[track_id]
            state.previous_center = state.center
            state.center = detection.bbox.center
            state.bbox = detection.bbox
            state.confidence = detection.confidence
            state.last_frame = frame_index
            state.missed = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in unmatched_tracks:
            self._states[track_id].missed += 1
        for track_id in [key for key, state in self._states.items() if state.missed > self.max_lost_frames]:
            del self._states[track_id]

        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            center = detection.bbox.center
            self._states[self._next_id] = _State(
                self._next_id, detection.bbox, center, center, detection.confidence, frame_index
            )
            self._next_id += 1

        return [
            Track(state.track_id, state.bbox, state.center, timestamp_s, state.confidence)
            for state in self._states.values()
            if state.missed == 0
        ]
