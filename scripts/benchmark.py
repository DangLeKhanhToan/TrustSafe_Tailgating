from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create benchmark metadata from a completed run")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--elapsed-s", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 0 or args.elapsed_s <= 0:
        parser.error("frames must be non-negative and elapsed-s must be positive")
    payload = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "model": args.model,
        "input_size": args.input_size,
        "frames": args.frames,
        "elapsed_s": args.elapsed_s,
        "end_to_end_fps": args.frames / args.elapsed_s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
