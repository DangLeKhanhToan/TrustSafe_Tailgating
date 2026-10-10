from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tailgating.pipeline import run_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Offline tailgating detection")
    result.add_argument("--video", type=Path, required=True)
    result.add_argument("--grants", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--model", type=Path)
    result.add_argument("--gt", type=Path, help="Ground truth CSV; run evaluate.py after inference")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        result = run_pipeline(args.video, args.grants, args.config, args.out, args.model)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"frames={result.frames} elapsed_s={result.elapsed_s:.3f} fps={result.fps:.2f}")
    print(args.out / "events.json")
    print(args.out / "annotated.mp4")
    if args.gt:
        from evaluate import evaluate_files
        metrics_path = args.out / "metrics.json"
        evaluate_files(args.out / "events.json", args.grants, args.gt, metrics_path)
        print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
