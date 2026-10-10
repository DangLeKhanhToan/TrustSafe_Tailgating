from __future__ import annotations

import unittest

from tailgating.events import EventAggregator
from tailgating.schemas import Crossing, Grant


class EventAggregationTests(unittest.TestCase):
    def test_same_track_counts_once_per_grant(self) -> None:
        aggregator = EventAggregator([Grant("g01", 10, 15)], 0, 0.5)
        aggregator.add(Crossing(7, 11, "in"))
        aggregator.add(Crossing(7, 13, "in"))
        event = aggregator.finalize()[0]
        self.assertEqual(event.entries, 1)
        self.assertEqual(event.alert, "none")

    def test_two_tracks_raise_tailgating_alert(self) -> None:
        aggregator = EventAggregator([Grant("g01", 10, 15)], 0, 0.5)
        aggregator.add(Crossing(7, 11, "in"))
        aggregator.add(Crossing(8, 12, "in"))
        event = aggregator.finalize()[0]
        self.assertEqual(event.entries, 2)
        self.assertEqual(event.alert, "tailgating")

    def test_exit_is_not_counted_and_unused_grant_is_kept(self) -> None:
        aggregator = EventAggregator([Grant("g01", 10, 15)], 0, 0.5)
        aggregator.add(Crossing(7, 11, "out"))
        self.assertEqual(aggregator.finalize()[0].entries, 0)

    def test_unmatched_entry_is_unauthorised(self) -> None:
        aggregator = EventAggregator([Grant("g01", 10, 15)], 0, 0.5)
        aggregator.add(Crossing(7, 20, "in"))
        events = aggregator.finalize()
        self.assertIsNone(events[1].grant_id)
        self.assertEqual(events[1].alert, "unauthorised_entry")

    def test_status_reports_active_grant_count_and_alert(self) -> None:
        aggregator = EventAggregator([Grant("g01", 10, 15)], 0, 0.5)
        aggregator.add(Crossing(7, 11, "in"))
        aggregator.add(Crossing(8, 12, "in"))
        self.assertEqual(aggregator.status_at(12.5), ("g01", 2, "tailgating"))
