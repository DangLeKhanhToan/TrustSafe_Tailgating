from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailgating.batch import BatchJob, discover_jobs, summarize_rows, write_rows


RUNTIME_PATTERN = re.compile(r"frames=(\d+) elapsed_s=([0-9.]+) fps=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively infer and benchmark a labeled clips tree")
    parser.add_argument("--clips", type=Path, default=Path("clips"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--results", type=Path, default=Path("results/baseline_416"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def completed_row(job: BatchJob, out_dir: Path, stdout: str) -> dict:
    match = RUNTIME_PATTERN.search(stdout)
    if match is None:
        raise ValueError("detector output did not contain runtime summary")
    frames, elapsed_s, fps = int(match.group(1)), float(match.group(2)), float(match.group(3))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    confusion = metrics["confusion"]
    grants = int(metrics.get("grant_count", 0))
    unauthorized = metrics.get("unauthorized", {})
    unauthorized_confusion = unauthorized.get("confusion", {})
    return {
        "session": job.session, "clip": job.stem, "scenario_id": job.scenario_id, "status": "ok",
        "frames": frames, "elapsed_s": elapsed_s, "fps": fps,
        "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        "count_accuracy": metrics["count_accuracy"], "count_mae": metrics["count_mae"],
        "tp": confusion["tp"], "fp": confusion["fp"], "fn": confusion["fn"], "tn": confusion["tn"],
        "exact": metrics["count_accuracy"] * grants, "grants": grants,
        "unauthorized_gt": unauthorized.get("ground_truth", 0),
        "unauthorized_pred": unauthorized.get("predicted", 0),
        "unauthorized_tp": unauthorized_confusion.get("tp", 0),
        "unauthorized_fp": unauthorized_confusion.get("fp", 0),
        "unauthorized_fn": unauthorized_confusion.get("fn", 0),
        "error": "",
    }


def main() -> int:
    args = parse_args()
    jobs, discovery_errors = discover_jobs(args.clips, args.configs)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"discovered={len(jobs)} invalid={len(discovery_errors)}", flush=True)
    for error in discovery_errors:
        print(f"dataset error: {error}", file=sys.stderr)
    if not jobs:
        return 2
    rows: list[dict] = []
    args.results.mkdir(parents=True, exist_ok=True)
    benchmark_csv = args.results / "metrics_by_clip.csv"
    for index, job in enumerate(jobs, start=1):
        out_dir = args.results / job.session / job.stem
        print(f"[{index}/{len(jobs)}] {job.session}/{job.stem}", flush=True)
        command = [
            sys.executable, str(ROOT / "detect_tailgating.py"),
            "--video", str(job.video), "--grants", str(job.grants), "--gt", str(job.ground_truth),
            "--config", str(job.config), "--model", str(args.model), "--out", str(out_dir),
        ]
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        wall_s = time.perf_counter() - started
        if process.returncode == 0:
            try:
                row = completed_row(job, out_dir, process.stdout)
                row["wall_s"] = wall_s
                print(f"  fps={row['fps']:.2f} count_accuracy={row['count_accuracy']:.3f}", flush=True)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                row = {"session": job.session, "clip": job.stem, "scenario_id": job.scenario_id, "status": "failed", "error": str(exc)}
        else:
            message = (process.stderr or process.stdout).strip().replace("\n", " | ")
            row = {"session": job.session, "clip": job.stem, "scenario_id": job.scenario_id, "status": "failed", "error": message}
            print(f"  FAILED: {message}", file=sys.stderr, flush=True)
        rows.append(row)
        write_rows(benchmark_csv, rows)

    by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_scenario[int(row["scenario_id"])].append(row)
    summary = {
        "protocol": {
            "model": str(args.model), "model_input": "fixed 416x416",
            "tracker": "CentroidTracker", "frame_stride": 1, "render_annotated_video": True,
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "dataset": {
            "clips_discovered": len(jobs), "invalid_items": discovery_errors,
            "scenario_ids_present": sorted(by_scenario),
            "missing_required_scenarios": sorted(set(range(1, 11)) - set(by_scenario)),
        },
        "overall": summarize_rows(rows),
        "by_scenario": {str(key): summarize_rows(value) for key, value in sorted(by_scenario.items())},
        "failed": [row for row in rows if row.get("status") != "ok"],
    }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.results / "summary.json")
    return 1 if summary["failed"] or discovery_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
