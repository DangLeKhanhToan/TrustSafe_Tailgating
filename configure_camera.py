from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw a door threshold and choose the inside side")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import cv2
        import numpy as np
        import yaml
    except ImportError as exc:
        raise SystemExit(f"opencv-python and PyYAML are required: {exc}")
    capture = cv2.VideoCapture(str(args.video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"cannot read video: {args.video}")
    points: list[tuple[int, int]] = []
    window = "Door setup: click line endpoints, then INSIDE"

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = frame.copy()
        if points:
            for point in points:
                cv2.circle(view, point, 6, (0, 220, 255), -1)
        if len(points) >= 2:
            cv2.line(view, points[0], points[1], (0, 220, 255), 3)
        if len(points) == 3:
            yy, xx = np.indices(view.shape[:2])
            ax, ay = points[0]
            bx, by = points[1]
            signed = (bx - ax) * (yy - ay) - (by - ay) * (xx - ax)
            ref_signed = (bx - ax) * (points[2][1] - ay) - (by - ay) * (points[2][0] - ax)
            inside = signed * (1 if ref_signed > 0 else -1) > 12
            outside = signed * (1 if ref_signed > 0 else -1) < -12
            overlay = view.copy()
            overlay[inside] = (40, 150, 40)
            overlay[outside] = (150, 80, 30)
            view = cv2.addWeighted(overlay, 0.22, view, 0.78, 0)
            mid = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            cv2.arrowedLine(view, mid, points[2], (0, 220, 0), 3, tipLength=0.15)
            cv2.putText(view, "INSIDE", points[2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            outside_label = (max(10, 2 * mid[0] - points[2][0]), max(45, 2 * mid[1] - points[2][1]))
            cv2.putText(view, "OUTSIDE", outside_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)
        instruction = "Click A, B, INSIDE | U undo | R reset | S save | Esc quit"
        cv2.putText(view, instruction, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key in (ord("u"), ord("U")) and points:
            points.pop()
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (ord("s"), ord("S")) and len(points) == 3:
            payload = {
                "camera": {"id": args.video.stem},
                "door": {"threshold": [list(points[0]), list(points[1])], "inside_reference_point": list(points[2]), "hysteresis_px": 12},
                "crossing": {"anchor": "bbox_center", "stable_frames": 3, "min_displacement_px": 25},
                "grant": {"pre_grace_s": 0.0, "post_grace_s": 0.5},
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            print(args.out)
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
