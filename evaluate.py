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


def evaluate_files(events_path: Path, grants_path: Path, gt_path: Path, out_path: Path) -> dict:
    grants = load_grants(grants_path)
    predictions = json.loads(events_path.read_text(encoding="utf-8"))
    predicted = {row["grant_id"]: int(row["entries"]) for row in predictions if row["grant_id"] is not None}
    counts: Counter[str] = Counter()
    with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["direction"] == "in" and row.get("grant_id"):
                counts[row["grant_id"]] += 1
    metrics = evaluate_counts([counts[g.grant_id] for g in grants], [predicted.get(g.grant_id, 0) for g in grants])
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
