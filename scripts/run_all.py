from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for every grants CSV in a clips directory")
    parser.add_argument("--clips", type=Path, default=Path("clips"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    failures = 0
    for grants in sorted(args.clips.glob("*_grants.csv")):
        stem = grants.name.removesuffix("_grants.csv")
        video = args.clips / f"{stem}.mp4"
        config = args.configs / f"{stem}.yaml"
        if not video.is_file() or not config.is_file():
            print(f"missing video/config for {stem}", file=sys.stderr)
            failures += 1
            continue
        command = [
            sys.executable, "detect_tailgating.py", "--video", str(video), "--grants", str(grants),
            "--config", str(config), "--model", str(args.model), "--out", str(args.results / stem),
        ]
        failures += int(subprocess.run(command, check=False).returncode != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
