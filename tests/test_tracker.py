from __future__ import annotations

import unittest

from tailgating.schemas import BoundingBox, Detection
from tailgating.tracker import CentroidTracker


class TrackerTests(unittest.TestCase):
    def test_preserves_id_across_steady_motion_and_one_missing_frame(self) -> None:
        tracker = CentroidTracker(max_lost_frames=2, max_center_distance=40)
        first = tracker.update([Detection(BoundingBox(10, 10, 30, 50), 0.9)], 0, 0.0)
        tracker.update([], 1, 0.1)
        third = tracker.update([Detection(BoundingBox(20, 10, 40, 50), 0.9)], 2, 0.2)
        self.assertEqual(first[0].track_id, third[0].track_id)

    def test_expires_track_after_lost_limit(self) -> None:
        tracker = CentroidTracker(max_lost_frames=1, max_center_distance=20)
        first = tracker.update([Detection(BoundingBox(10, 10, 30, 50), 0.9)], 0, 0.0)
        tracker.update([], 1, 0.1)
        tracker.update([], 2, 0.2)
        later = tracker.update([Detection(BoundingBox(10, 10, 30, 50), 0.9)], 3, 0.3)
        self.assertNotEqual(first[0].track_id, later[0].track_id)

    def test_two_people_receive_distinct_ids(self) -> None:
        tracker = CentroidTracker(max_lost_frames=2, max_center_distance=30)
        tracks = tracker.update([
            Detection(BoundingBox(10, 10, 30, 50), 0.9),
            Detection(BoundingBox(100, 10, 120, 50), 0.8),
        ], 0, 0.0)
        self.assertEqual(len({track.track_id for track in tracks}), 2)
