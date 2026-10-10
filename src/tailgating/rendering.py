from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any

from .geometry import entry_normal
from .schemas import CameraConfig, Track


class Renderer:
    def __init__(self, path: Path, fps: float, width: int, height: int, config: CameraConfig) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for rendering") from exc
        self.cv2 = cv2
        self.config = config
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create annotated video: {path}")

    def draw(
        self, frame: Any, tracks: Sequence[Track], timestamp_s: float,
        grant_id: str | None = None, entries: int = 0, alert: str = "none",
    ) -> Any:
        cv2 = self.cv2
        result = frame.copy()
        start = tuple(map(round, self.config.threshold.start))
        end = tuple(map(round, self.config.threshold.end))
        cv2.line(result, start, end, (0, 220, 255), 3)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        normal = entry_normal(self.config.threshold)
        arrow = (round(mid[0] + normal[0] * 70), round(mid[1] + normal[1] * 70))
        cv2.arrowedLine(result, mid, arrow, (0, 220, 0), 3, tipLength=0.25)
        cv2.putText(result, "IN", arrow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        for track in tracks:
            box = track.bbox
            p1, p2 = (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2))
            cv2.rectangle(result, p1, p2, (255, 160, 0), 2)
            cv2.circle(result, (round(track.anchor.x), round(track.anchor.y)), 4, (0, 0, 255), -1)
            cv2.putText(result, f"ID {track.track_id}", (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
        cv2.putText(result, f"t={timestamp_s:.2f}s", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"grant={grant_id or '-'} entries={entries}", (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if alert != "none":
            cv2.rectangle(result, (0, 70), (result.shape[1], 125), (0, 0, 190), -1)
            cv2.putText(result, alert.upper(), (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return result

    def write(
        self, frame: Any, tracks: Sequence[Track], timestamp_s: float,
        grant_id: str | None = None, entries: int = 0, alert: str = "none",
    ) -> None:
        self.writer.write(self.draw(frame, tracks, timestamp_s, grant_id, entries, alert))

    def close(self) -> None:
        self.writer.release()
