from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tailgating.annotation import AnnotationDocument
from tailgating.cutting import plan_event_cut, quantize_cut_to_frames, rebase_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut labeled events into benchmark clips")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--padding-s", type=float, default=5.0)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    document = AnnotationDocument.load(args.annotations)
    document.validate()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise SystemExit("video has invalid FPS, frame count, or dimensions")
    try:
        document.validate_source(
            str(args.video), document.data["camera_id"], fps, frame_count, width, height
        )
        print(f"video: {args.video} ({fps:.3f} FPS, {frame_count} frames, {width}x{height})")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    duration_s = frame_count / fps
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"cutting {len(document.data['events'])} events from {args.video} to {args.out}")
    for event in document.data["events"]:
        print(f"cutting event {event['event_id']} ({event['scenario_id']})")
        start_s, end_s = plan_event_cut(event, duration_s, args.padding_s)
        start_frame, end_frame, actual_start_s, actual_end_s = quantize_cut_to_frames(
            start_s, end_s, fps, frame_count
        )
        grants, crossings = rebase_annotations(
            document.data["grants"], document.data["crossings"], actual_start_s, actual_end_s
        )
        stem = f"{document.data['session_id']}_{event['event_id']}"
        # print(f"cutting {stem}: {actual_start_s:.3f}s to {actual_end_s:.3f}s ({start_frame} to {end_frame})")
        video_path = args.out / f"{stem}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"cannot create output video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        write_csv(args.out / f"{stem}_grants.csv", ["grant_id", "t_unlock_s", "t_relock_s"], grants)
        write_csv(
            args.out / f"{stem}_gt.csv",
            ["crossing_id", "person_id", "t_cross_s", "direction", "grant_id"],
            crossings,
        )
        metadata = {
            "source_session_id": document.data["session_id"],
            "source_video": str(args.video),
            "source_start_s": round(actual_start_s, 6),
            "source_end_s": round(actual_end_s, 6),
            "event_id": event["event_id"],
            "scenario_id": event["scenario_id"],
        }
        (args.out / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(video_path)
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
