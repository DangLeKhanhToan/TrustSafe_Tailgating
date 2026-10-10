from __future__ import annotations

from typing import Sequence


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_counts(ground_truth: Sequence[int], predicted: Sequence[int]) -> dict:
    if len(ground_truth) != len(predicted):
        raise ValueError("ground-truth and predicted counts must have equal length")
    tp = fp = fn = tn = exact = 0
    absolute_error = 0
    for expected, actual in zip(ground_truth, predicted):
        gt_alert, pred_alert = expected > 1, actual > 1
        tp += int(gt_alert and pred_alert)
        fp += int(not gt_alert and pred_alert)
        fn += int(gt_alert and not pred_alert)
        tn += int(not gt_alert and not pred_alert)
        exact += int(expected == actual)
        absolute_error += abs(expected - actual)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "count_accuracy": _safe_divide(exact, len(ground_truth)),
        "count_mae": _safe_divide(absolute_error, len(ground_truth)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
