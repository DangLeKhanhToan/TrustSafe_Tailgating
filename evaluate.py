from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tailgating.grants import load_grants
from tailgating.metrics import evaluate_counts


def load_ground_truth(gt_path: Path) -> tuple[Counter[str], list[float]]:
    people_by_grant: dict[str, set[str]] = {}
    unauthorized: list[float] = []
    with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("direction") != "in":
                continue
            person_id = (row.get("person_id") or "").strip()
            grant_id = (row.get("grant_id") or "").strip()
            if not person_id:
                raise ValueError(f"missing person_id in ground truth row {row_number}")
            if grant_id:
                people_by_grant.setdefault(grant_id, set()).add(person_id)
            else:
                unauthorized.append(float(row["t_cross_s"]))
    return Counter({grant_id: len(people) for grant_id, people in people_by_grant.items()}), unauthorized


def match_timestamps(expected: list[float], predicted: list[float], tolerance_s: float = 1.0) -> dict[str, int]:
    candidates = sorted(
        (abs(gt - pred), gt_index, pred_index)
        for gt_index, gt in enumerate(expected)
        for pred_index, pred in enumerate(predicted)
        if abs(gt - pred) <= tolerance_s
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return {"tp": len(matched_gt), "fp": len(predicted) - len(matched_pred), "fn": len(expected) - len(matched_gt)}


def evaluate_files(events_path: Path, grants_path: Path, gt_path: Path, out_path: Path) -> dict:
    grants = load_grants(grants_path)
    predictions = json.loads(events_path.read_text(encoding="utf-8"))
    predicted: dict[str, int] = {}
    known_grants = {grant.grant_id for grant in grants}
    for row in predictions:
        grant_id = row.get("grant_id")
        if grant_id is None:
            continue
        if grant_id not in known_grants:
            raise ValueError(f"prediction references unknown grant_id: {grant_id}")
        if grant_id in predicted:
            raise ValueError(f"duplicate prediction for grant_id: {grant_id}")
        predicted[grant_id] = int(row["entries"])
    counts, gt_unauthorized = load_ground_truth(gt_path)
    metrics = evaluate_counts([counts[g.grant_id] for g in grants], [predicted.get(g.grant_id, 0) for g in grants])
    predicted_unauthorized = [
        float(crossing["t_cross_s"])
        for row in predictions
        if row.get("grant_id") is None and row.get("alert") == "unauthorised_entry"
        for crossing in row.get("crossings", [])
    ]
    unauthorized_confusion = match_timestamps(gt_unauthorized, predicted_unauthorized)
    metrics["grant_count"] = len(grants)
    metrics["unauthorized"] = {
        "ground_truth": len(gt_unauthorized),
        "predicted": len(predicted_unauthorized),
        "tolerance_s": 1.0,
        "confusion": unauthorized_confusion,
        "exact": unauthorized_confusion["fp"] == 0 and unauthorized_confusion["fn"] == 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted grant counts")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--grants", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate_files(args.events, args.grants, args.gt, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
