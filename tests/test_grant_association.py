from __future__ import annotations

import unittest

from tailgating.grants import find_grant
from tailgating.schemas import Crossing, Grant


class GrantAssociationTests(unittest.TestCase):
    def test_effective_grant_boundaries_are_inclusive(self) -> None:
        grant = Grant("g01", 10.0, 15.0)
        for timestamp in (10.0, 15.5):
            with self.subTest(timestamp=timestamp):
                crossing = Crossing(1, timestamp, "in")
                self.assertEqual(find_grant(crossing, [grant], 0.0, 0.5), grant)

    def test_overlap_selects_latest_unlocked_grant(self) -> None:
        grants = [Grant("g01", 10, 15), Grant("g02", 14, 19)]
        crossing = Crossing(1, 14.5, "in")
        self.assertEqual(find_grant(crossing, grants, 0, 0).grant_id, "g02")
