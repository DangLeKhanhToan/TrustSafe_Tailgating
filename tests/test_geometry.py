from __future__ import annotations

import unittest

from tailgating.geometry import Zone, classify_zone, crossing_time
from tailgating.schemas import ThresholdConfig


def threshold(reverse: bool = False) -> ThresholdConfig:
    start, end = ((50.0, 0.0), (50.0, 100.0))
    if reverse:
        start, end = end, start
    return ThresholdConfig(start, end, (80.0, 50.0), 5.0)


class GeometryTests(unittest.TestCase):
    def test_vertical_threshold_uses_reference_side_as_inside(self) -> None:
        self.assertIs(classify_zone((40.0, 50.0), threshold()), Zone.OUTSIDE)
        self.assertIs(classify_zone((52.0, 50.0), threshold()), Zone.BAND)
        self.assertIs(classify_zone((70.0, 50.0), threshold()), Zone.INSIDE)

    def test_reversing_endpoints_preserves_reference_defined_inside(self) -> None:
        self.assertIs(classify_zone((70.0, 50.0), threshold()), Zone.INSIDE)
        self.assertIs(classify_zone((70.0, 50.0), threshold(True)), Zone.INSIDE)

    def test_crossing_time_interpolates_line_intersection(self) -> None:
        value = crossing_time((40.0, 50.0), 1.0, (60.0, 50.0), 2.0, threshold())
        self.assertAlmostEqual(value, 1.5)
