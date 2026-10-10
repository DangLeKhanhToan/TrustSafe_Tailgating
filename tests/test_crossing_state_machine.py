from __future__ import annotations

import unittest

from tailgating.crossing import CrossingDetector
from tailgating.schemas import Point, ThresholdConfig, Track


def make_detector() -> CrossingDetector:
    return CrossingDetector(
        ThresholdConfig((50, 0), (50, 100), (80, 50), 5),
        stable_frames=2,
        min_displacement_px=12,
    )


def feed(detector: CrossingDetector, xs: list[float], track_id: int = 7):
    events = []
    for index, x in enumerate(xs):
        track = Track(track_id, (x - 5, 20, x + 5, 80), Point(x, 50), index * 0.1)
        event = detector.update(track)
        if event:
            events.append(event)
    return events


class CrossingTests(unittest.TestCase):
    def test_outside_to_inside_emits_one_entry(self) -> None:
        events = feed(make_detector(), [30, 35, 45, 52, 60, 67])
        self.assertEqual([(event.track_id, event.direction) for event in events], [(7, "in")])

    def test_band_jitter_does_not_emit_crossing(self) -> None:
        self.assertEqual(feed(make_detector(), [40, 48, 52, 49, 51, 45, 40]), [])

    def test_track_first_seen_inside_is_not_an_entry(self) -> None:
        self.assertEqual(feed(make_detector(), [65, 70, 75]), [])

    def test_hesitation_then_return_outside_is_not_an_entry(self) -> None:
        self.assertEqual(feed(make_detector(), [30, 35, 46, 50, 52, 44, 35]), [])
