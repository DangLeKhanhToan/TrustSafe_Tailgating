from __future__ import annotations

import unittest

from tailgating.pipeline import process_track_frames
from tailgating.schemas import Grant, Point, ThresholdConfig, Track


def track(track_id: int, x: float, timestamp: float) -> Track:
    return Track(track_id, (x - 5, 20, x + 5, 80), Point(x, 50), timestamp)


class DomainPipelineIntegrationTests(unittest.TestCase):
    def test_two_tracks_crossing_during_one_grant_raise_tailgating(self) -> None:
        xs = [30, 35, 45, 52, 60, 67]
        frames = []
        for index, x in enumerate(xs):
            timestamp = 10.0 + index * 0.1
            frames.append([track(1, x, timestamp), track(2, x - 2, timestamp)])
        events = process_track_frames(
            frames,
            [Grant("g01", 10.0, 12.0)],
            ThresholdConfig((50, 0), (50, 100), (80, 50), 5),
            stable_frames=2,
            min_displacement_px=12,
            pre_grace_s=0,
            post_grace_s=0.5,
        )
        self.assertEqual(events[0].entries, 2)
        self.assertEqual(events[0].alert, "tailgating")

    def test_crossing_outside_grant_is_unauthorised(self) -> None:
        frames = [[track(3, x, 20 + i * 0.1)] for i, x in enumerate([30, 35, 45, 52, 60, 67])]
        events = process_track_frames(
            frames,
            [Grant("g01", 10, 12)],
            ThresholdConfig((50, 0), (50, 100), (80, 50), 5),
            2, 12, 0, 0.5,
        )
        self.assertEqual(events[1].alert, "unauthorised_entry")
