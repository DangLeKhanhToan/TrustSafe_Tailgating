from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluate import evaluate_files
from tailgating.batch import summarize_rows


def flatten(group: str, key: str, summary: dict) -> dict:
    confusion = summary["confusion"]
    unauthorized = summary["unauthorized"]
    unauthorized_confusion = unauthorized["confusion"]
    return {
        group: key,
        "clips": summary["clips_total"],
        "grants": summary["grants"],
        "frames": summary["frames"],
        "elapsed_s": round(summary["elapsed_s"], 3),
        "fps": round(summary["processing_fps"], 3),
        "tailgating_precision": round(summary["precision"], 4),
        "tailgating_recall": round(summary["recall"], 4),
        "tailgating_f1": round(summary["f1"], 4),
        "count_accuracy": round(summary["count_accuracy"], 4),
        "count_mae": round(summary["count_mae"], 4),
        "tp": confusion["tp"], "fp": confusion["fp"], "fn": confusion["fn"], "tn": confusion["tn"],
        "unauthorized_precision": round(unauthorized["precision"], 4),
        "unauthorized_recall": round(unauthorized["recall"], 4),
        "unauthorized_f1": round(unauthorized["f1"], 4),
        "unauthorized_tp": unauthorized_confusion["tp"],
        "unauthorized_fp": unauthorized_confusion["fp"],
        "unauthorized_fn": unauthorized_confusion["fn"],
    }


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate a completed batch benchmark")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--clips", type=Path, default=Path("clips"))
    args = parser.parse_args()
    with (args.results / "metrics_by_clip.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("metrics_by_clip.csv is empty")
    for row in rows:
        session, clip = row["session"], row["clip"]
        result_dir = args.results / session / clip
        metrics = evaluate_files(
            result_dir / "events.json",
            args.clips / session / f"{clip}_grants.csv",
            args.clips / session / f"{clip}_gt.csv",
            result_dir / "metrics.json",
        )
        confusion = metrics["confusion"]
        unauthorized = metrics["unauthorized"]
        unauthorized_confusion = unauthorized["confusion"]
        row.update({
            "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
            "count_accuracy": metrics["count_accuracy"], "count_mae": metrics["count_mae"],
            "tp": confusion["tp"], "fp": confusion["fp"], "fn": confusion["fn"], "tn": confusion["tn"],
            "exact": metrics["count_accuracy"] * metrics["grant_count"], "grants": metrics["grant_count"],
            "unauthorized_gt": unauthorized["ground_truth"], "unauthorized_pred": unauthorized["predicted"],
            "unauthorized_tp": unauthorized_confusion["tp"],
            "unauthorized_fp": unauthorized_confusion["fp"],
            "unauthorized_fn": unauthorized_confusion["fn"],
        })
    from tailgating.batch import write_rows
    write_rows(args.results / "metrics_by_clip.csv", rows)
    by_session: dict[str, list[dict]] = defaultdict(list)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row["session"]].append(row)
        by_scenario[row["scenario_id"]].append(row)
    overall = summarize_rows(rows)
    session_rows = [flatten("session", key, summarize_rows(value)) for key, value in sorted(by_session.items())]
    scenario_rows = [flatten("scenario_id", key, summarize_rows(value)) for key, value in sorted(by_scenario.items(), key=lambda item: int(item[0]))]
    write_table(args.results / "metrics_by_session.csv", session_rows)
    write_table(args.results / "metrics_by_scenario.csv", scenario_rows)
    summary_path = args.results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        import onnxruntime
        onnxruntime_version = onnxruntime.__version__
    except ImportError:
        onnxruntime_version = "unknown"
    summary["protocol"]["processor_identifier"] = os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    summary["protocol"]["onnxruntime"] = onnxruntime_version
    summary["protocol"]["speed_definition"] = "total processed frames / summed pipeline elapsed_s; excludes process and model startup"
    summary["overall"] = overall
    summary["by_session"] = {row["session"]: summarize_rows(by_session[row["session"]]) for row in session_rows}
    summary["by_scenario"] = {row["scenario_id"]: summarize_rows(by_scenario[row["scenario_id"]]) for row in scenario_rows}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
