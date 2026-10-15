import cv2
import numpy as np

from tailgating.crossing import CrossingDebugState
from tailgating.rendering import Renderer, prioritize_debug_states
from tailgating.schemas import CameraConfig, ThresholdConfig


def make_renderer() -> Renderer:
    renderer = Renderer.__new__(Renderer)
    renderer.cv2 = cv2
    renderer.config = CameraConfig(
        camera_id="test",
        threshold=ThresholdConfig((20, 0), (20, 80), (40, 40), 5),
    )
    renderer.header_height = 140
    return renderer


def test_draw_adds_black_header_without_resizing_source_frame():
    frame = np.full((80, 120, 3), 100, dtype=np.uint8)

    result = make_renderer().draw(frame, [], 1.25)

    assert result.shape == (220, 120, 3)
    assert np.all(result[0, 0] == 0)


def test_draw_accepts_up_to_three_candidate_states():
    states = [
        CrossingDebugState(index, "OUTSIDE", "ARMED", 3, 0, 3)
        for index in range(1, 4)
    ]

    result = make_renderer().draw(np.zeros((80, 500, 3), dtype=np.uint8), [], 1.0, debug_states=states)

    assert result.shape == (220, 500, 3)


def test_alert_does_not_replace_active_grant_led():
    frame = np.zeros((80, 1000, 3), dtype=np.uint8)

    result = make_renderer().draw(frame, [], 1.0, grant_id="g001", entries=2, alert="tailgating")

    assert tuple(result[28, 25]) == (0, 190, 0)
    assert tuple(result[10, 340]) == (0, 0, 210)


def test_transition_candidates_are_prioritized_over_low_track_ids():
    states = [
        CrossingDebugState(1, "INSIDE", "INSIDE", 3, 8, 3),
        CrossingDebugState(2, "OUTSIDE", "OBSERVING", 1, 0, 3),
        CrossingDebugState(3, "OUTSIDE", "ARMED", 3, 0, 3),
        CrossingDebugState(9, "BAND", "IN_BAND", 3, 0, 3),
    ]

    selected = prioritize_debug_states(states, limit=3)

    assert [state.track_id for state in selected] == [9, 3, 2]
