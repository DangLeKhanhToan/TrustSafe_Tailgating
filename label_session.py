from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tailgating.annotation import AnnotationDocument
from tailgating.schemas import load_camera_config


HELP = "SPACE play | A/D frame | J/L 5s | G/R grant | B/E event | 1-3 person | I/O crossing | S save | U undo | Q quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight tailgating session labeler")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_camera_config(args.camera_config)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        raise SystemExit("video has invalid FPS or frame count")
    if args.out.exists():
        document = AnnotationDocument.load(args.out)
        try:
            document.validate_source(str(args.video), config.camera_id, fps, frame_count, width, height)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Resuming {args.out}")
    else:
        document = AnnotationDocument.new(
            args.video.stem,
            str(args.video),
            config.camera_id,
            fps,
            frame_count,
            width,
            height,
            config.pre_grace_s,
            config.post_grace_s,
        )
    frame_index = 0
    playing = False
    person_number = 1
    scenario_id = 1
    status = "Ready"
    window = "Tailgating labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            frame_index = max(0, frame_count - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            playing = False
        if not ok:
            break
        timestamp_s = frame_index / fps
        start = tuple(map(int, config.threshold.start))
        end = tuple(map(int, config.threshold.end))
        reference = tuple(map(int, config.threshold.inside_reference_point))
        cv2.line(frame, start, end, (0, 220, 255), 3)
        cv2.circle(frame, reference, 8, (0, 220, 0), -1)
        grant = document.active_grant(timestamp_s)
        led_color = (0, 220, 0) if grant else (80, 80, 80)
        led_text = f"GRANT {grant['grant_id']} ACTIVE" if grant else "DOOR LOCKED"
        cv2.circle(frame, (30, 35), 12, led_color, -1)
        cv2.putText(frame, led_text, (50, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.75, led_color, 2)
        rows = [
            f"{timestamp_s:07.2f}s  frame {frame_index}/{frame_count - 1}",
            f"person p{person_number:03d}  scenario {scenario_id}",
            f"grants {len(document.data['grants'])}  crossings {len(document.data['crossings'])}  events {len(document.data['events'])}",
            status,
            HELP,
        ]
        for row, text in enumerate(rows):
            y = height - 105 + row * 23
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(window, frame)
        delay = max(1, int(1000 / fps)) if playing else 0
        key = cv2.waitKeyEx(delay)
        if playing and key == -1:
            frame_index = min(frame_index + 1, frame_count - 1)
            if frame_index == frame_count - 1:
                playing = False
            continue
        if key in (-1,):
            continue
        try:
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                playing = not playing
            elif key in (ord("a"), ord("A"), 2424832):
                frame_index = max(0, frame_index - 1)
            elif key in (ord("d"), ord("D"), 2555904):
                frame_index = min(frame_count - 1, frame_index + 1)
            elif key in (ord("j"), ord("J")):
                frame_index = max(0, frame_index - round(5 * fps))
            elif key in (ord("l"), ord("L")):
                frame_index = min(frame_count - 1, frame_index + round(5 * fps))
            elif key in (ord("1"), ord("2"), ord("3")):
                person_number = int(chr(key))
                status = f"Selected p{person_number:03d}"
            elif key in (ord("g"), ord("G")):
                item = document.start_grant(timestamp_s)
                status = f"Started {item['grant_id']}"
            elif key in (ord("r"), ord("R")):
                item = document.close_grant(timestamp_s)
                status = f"Closed {item['grant_id']}"
            elif key in (ord("b"), ord("B")):
                item = document.start_event(timestamp_s, scenario_id)
                status = f"Started {item['event_id']} scenario {scenario_id}"
            elif key in (ord("e"), ord("E")):
                item = document.close_event(timestamp_s)
                status = f"Closed {item['event_id']}"
            elif key in (ord("i"), ord("I"), ord("o"), ord("O")):
                direction = "in" if chr(key).lower() == "i" else "out"
                item = document.add_crossing(timestamp_s, direction, f"p{person_number:03d}")
                status = f"Added {item['crossing_id']} {direction}; grant={item['grant_id']}"
            elif key in (ord("u"), ord("U")):
                status = "Undo" if document.undo() else "Nothing to undo"
            elif key in (ord("s"), ord("S")):
                document.save(args.out)
                status = f"Saved {args.out}"
                print(status)
        except ValueError as exc:
            status = f"ERROR: {exc}"
        if key in (ord("+"), ord("=")):
            scenario_id = min(99, scenario_id + 1)
        elif key in (ord("-"), ord("_")):
            scenario_id = max(1, scenario_id - 1)

    capture.release()
    cv2.destroyAllWindows()
    if document.open_grant() is not None or document.open_event() is not None:
        print("Not saved: close open grant/event first, then press S")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
