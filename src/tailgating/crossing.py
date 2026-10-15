from __future__ import annotations

from dataclasses import dataclass

from .geometry import Zone, classify_zone, crossing_time, normalized_distance
from .schemas import Crossing, Point, ThresholdConfig, Track


@dataclass
class _TrackState:
    zone: Zone
    outside_seen: int = 0
    inside_seen: int = 0
    transition_origin: Point | None = None
    previous_point: Point | None = None
    previous_time: float = 0.0
    candidate_time: float | None = None
    entry_emitted: bool = False


@dataclass(frozen=True)
class CrossingDebugState:
    track_id: int
    zone: str
    stage: str
    outside_seen: int
    inside_seen: int
    stable_frames: int


class CrossingDetector:
    def __init__(self, threshold: ThresholdConfig, stable_frames: int = 3, min_displacement_px: float = 25) -> None:
        if stable_frames <= 0 or min_displacement_px <= 0:
            raise ValueError("crossing settings must be positive")
        self.threshold = threshold
        self.stable_frames = stable_frames
        self.min_displacement_px = min_displacement_px
        self._states: dict[int, _TrackState] = {}

    def debug_state(self, track_id: int) -> CrossingDebugState:
        state = self._states[track_id]
        if state.entry_emitted:
            stage = "CROSSED"
        elif state.zone is Zone.BAND:
            stage = "IN_BAND"
        elif state.candidate_time is not None:
            stage = "CONFIRMING"
        elif state.zone is Zone.OUTSIDE and state.outside_seen >= self.stable_frames:
            stage = "ARMED"
        elif state.zone is Zone.INSIDE:
            stage = "INSIDE"
        else:
            stage = "OBSERVING"
        return CrossingDebugState(
            track_id,
            state.zone.value.upper(),
            stage,
            state.outside_seen,
            state.inside_seen,
            self.stable_frames,
        )

    def update(self, track: Track) -> Crossing | None:
        point = track.anchor
        zone = classify_zone(point.as_tuple(), self.threshold)
        state = self._states.get(track.track_id)
        if state is None:
            state = _TrackState(zone=zone, previous_point=point, previous_time=track.timestamp_s)
            state.outside_seen = 1 if zone is Zone.OUTSIDE else 0
            state.inside_seen = 1 if zone is Zone.INSIDE else 0
            self._states[track.track_id] = state
            return None

        if zone is Zone.OUTSIDE:
            state.outside_seen += 1
            state.inside_seen = 0
            state.transition_origin = point if state.outside_seen >= self.stable_frames else state.transition_origin
            state.candidate_time = None
            state.entry_emitted = False
        elif zone is Zone.BAND:
            state.inside_seen = 0
            if state.outside_seen >= self.stable_frames and state.transition_origin is None:
                state.transition_origin = state.previous_point
        else:
            state.inside_seen += 1
            if state.outside_seen >= self.stable_frames and state.candidate_time is None:
                if state.previous_point is not None:
                    state.candidate_time = crossing_time(
                        state.previous_point.as_tuple(), state.previous_time,
                        point.as_tuple(), track.timestamp_s, self.threshold,
                    )
            if state.transition_origin is not None and state.candidate_time is not None:
                displacement = normalized_distance(point.as_tuple(), self.threshold) - normalized_distance(
                    state.transition_origin.as_tuple(), self.threshold
                )
                if state.inside_seen >= self.stable_frames and displacement >= self.min_displacement_px and not state.entry_emitted:
                    state.entry_emitted = True
                    event = Crossing(track.track_id, state.candidate_time, "in")
                    state.previous_point, state.previous_time, state.zone = point, track.timestamp_s, zone
                    return event

        state.previous_point, state.previous_time, state.zone = point, track.timestamp_s, zone
        return None
