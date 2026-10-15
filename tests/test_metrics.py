from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tailgating.metrics import evaluate_counts
from evaluate import load_ground_truth, match_timestamps


class MetricsTests(unittest.TestCase):
    def test_tailgating_metrics_and_exact_count_accuracy(self) -> None:
        metrics = evaluate_counts([1, 2, 3], [1, 1, 3])
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 0, "fn": 1, "tn": 1})
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["count_accuracy"], 2 / 3)

    def test_empty_metrics_are_finite_zeroes(self) -> None:
        metrics = evaluate_counts([], [])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["count_accuracy"], 0.0)

    def test_ground_truth_counts_unique_people_per_grant(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / "repeated_person_gt.csv"
            fixture.write_text(
                "person_id,t_cross_s,direction,grant_id\n"
                "p01,1.0,in,g01\n"
                "p01,2.0,out,\n"
                "p01,3.0,in,g01\n"
                "p02,4.0,in,\n",
                encoding="utf-8",
            )
            counts, unauthorized = load_ground_truth(fixture)

        self.assertEqual(counts["g01"], 1)
        self.assertEqual(unauthorized, [4.0])

    def test_unauthorized_timestamps_use_one_to_one_matching(self) -> None:
        self.assertEqual(
            match_timestamps([1.0, 10.0], [1.5, 20.0], tolerance_s=1.0),
            {"tp": 1, "fp": 1, "fn": 1},
        )
