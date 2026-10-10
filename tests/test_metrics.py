from __future__ import annotations

import unittest

from tailgating.metrics import evaluate_counts


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
