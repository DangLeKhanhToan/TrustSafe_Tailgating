from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailgating.grants import find_overlaps, load_grants


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate clip/grant/ground-truth companion files")
    parser.add_argument("--clips", type=Path, default=Path("clips"))
    parser.add_argument("--post-grace-s", type=float, default=0.5)
    args = parser.parse_args()
    errors = 0
    for grants_path in sorted(args.clips.glob("*_grants.csv")):
        stem = grants_path.name.removesuffix("_grants.csv")
        video, gt = args.clips / f"{stem}.mp4", args.clips / f"{stem}_gt.csv"
        for expected in (video, gt):
            if not expected.is_file():
                print(f"ERROR {stem}: missing {expected.name}")
                errors += 1
        try:
            grants = load_grants(grants_path)
        except ValueError as exc:
            print(f"ERROR {stem}: {exc}")
            errors += 1
            continue
        known = {grant.grant_id for grant in grants}
        for left, right in find_overlaps(grants, 0, args.post_grace_s):
            print(f"WARNING {stem}: effective grants overlap: {left}, {right}")
        if gt.is_file():
            with gt.open("r", encoding="utf-8-sig", newline="") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    if row.get("direction") not in ("in", "out"):
                        print(f"ERROR {stem}:{row_number}: invalid direction")
                        errors += 1
                    grant_id = (row.get("grant_id") or "").strip()
                    if grant_id and grant_id not in known:
                        print(f"ERROR {stem}:{row_number}: unknown grant_id {grant_id}")
                        errors += 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
