from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BatchJob:
    stem: str
    session: str
    scenario_id: int
    video: Path
    grants: Path
    ground_truth: Path
    metadata: Path
    config: Path


def discover_jobs(clips_root: Path, configs_root: Path) -> tuple[list[BatchJob], list[str]]:
    jobs: list[BatchJob] = []
    errors: list[str] = []
    for video in sorted(clips_root.rglob("*.mp4")):
        stem = video.stem
        grants = video.with_name(f"{stem}_grants.csv")
        ground_truth = video.with_name(f"{stem}_gt.csv")
        metadata = video.with_name(f"{stem}_metadata.json")
        missing = [path for path in (grants, ground_truth, metadata) if not path.is_file()]
        if missing:
            errors.append(f"{video}: missing " + ", ".join(str(path) for path in missing))
            continue
        try:
            metadata_data = json.loads(metadata.read_text(encoding="utf-8"))
            scenario_id = int(metadata_data["scenario_id"])
            session = str(metadata_data.get("source_session_id") or video.relative_to(clips_root).parts[0])
            config = configs_root / f"{session}.yaml"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{metadata}: invalid metadata: {exc}")
            continue
        if not config.is_file():
            errors.append(f"{video}: missing {config}")
            continue
        jobs.append(BatchJob(stem, session, scenario_id, video, grants, ground_truth, metadata, config))
    return jobs, errors


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    frames = sum(int(row.get("frames", 0)) for row in valid)
    elapsed_s = sum(float(row.get("elapsed_s", 0.0)) for row in valid)
    tp = sum(int(row.get("tp", 0)) for row in valid)
    fp = sum(int(row.get("fp", 0)) for row in valid)
    fn = sum(int(row.get("fn", 0)) for row in valid)
    tn = sum(int(row.get("tn", 0)) for row in valid)
    exact = sum(float(row.get("exact", 0.0)) for row in valid)
    grants = sum(int(row.get("grants", 0)) for row in valid)
    absolute_error = sum(float(row.get("count_mae", 0.0)) * int(row.get("grants", 0)) for row in valid)
    unauthorized_tp = sum(int(row.get("unauthorized_tp", 0)) for row in valid)
    unauthorized_fp = sum(int(row.get("unauthorized_fp", 0)) for row in valid)
    unauthorized_fn = sum(int(row.get("unauthorized_fn", 0)) for row in valid)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    unauthorized_precision = _safe_divide(unauthorized_tp, unauthorized_tp + unauthorized_fp)
    unauthorized_recall = _safe_divide(unauthorized_tp, unauthorized_tp + unauthorized_fn)
    return {
        "clips_total": len(valid),
        "frames": frames,
        "elapsed_s": elapsed_s,
        "processing_fps": _safe_divide(frames, elapsed_s),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "count_accuracy": _safe_divide(exact, grants),
        "count_mae": _safe_divide(absolute_error, grants),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "grants": grants,
        "unauthorized": {
            "precision": unauthorized_precision,
            "recall": unauthorized_recall,
            "f1": _safe_divide(
                2 * unauthorized_precision * unauthorized_recall,
                unauthorized_precision + unauthorized_recall,
            ),
            "confusion": {"tp": unauthorized_tp, "fp": unauthorized_fp, "fn": unauthorized_fn},
        },
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "session", "clip", "scenario_id", "status", "frames", "elapsed_s", "fps",
        "precision", "recall", "f1", "count_accuracy", "count_mae",
        "tp", "fp", "fn", "tn", "exact", "grants",
        "unauthorized_gt", "unauthorized_pred", "error",
        "unauthorized_tp", "unauthorized_fp", "unauthorized_fn", "wall_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
