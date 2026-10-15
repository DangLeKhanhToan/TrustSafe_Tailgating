from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any

from .geometry import entry_normal
from .crossing import CrossingDebugState
from .schemas import CameraConfig, Track


def prioritize_debug_states(
    states: Sequence[CrossingDebugState], limit: int = 3
) -> list[CrossingDebugState]:
    priority = {
        "IN_BAND": 0,
        "CONFIRMING": 1,
        "ARMED": 2,
        "OBSERVING": 3,
        "CROSSED": 4,
        "INSIDE": 5,
    }
    return sorted(states, key=lambda item: (priority.get(item.stage, 99), item.track_id))[:limit]


class Renderer:
    HEADER_HEIGHT = 140

    def __init__(self, path: Path, fps: float, width: int, height: int, config: CameraConfig) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for rendering") from exc
        self.cv2 = cv2
        self.config = config
        self.header_height = self.HEADER_HEIGHT
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + self.header_height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create annotated video: {path}")

    def draw(
        self, frame: Any, tracks: Sequence[Track], timestamp_s: float,
        grant_id: str | None = None, entries: int = 0, alert: str = "none",
        debug_states: Sequence[CrossingDebugState] = (),
    ) -> Any:
        cv2 = self.cv2
        video = frame.copy()
        start = tuple(map(round, self.config.threshold.start))
        end = tuple(map(round, self.config.threshold.end))
        cv2.line(video, start, end, (0, 220, 255), 3)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        normal = entry_normal(self.config.threshold)
        arrow = (round(mid[0] + normal[0] * 70), round(mid[1] + normal[1] * 70))
        cv2.arrowedLine(video, mid, arrow, (0, 220, 0), 3, tipLength=0.25)
        cv2.putText(video, "IN", arrow, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        for track in tracks:
            box = track.bbox
            p1, p2 = (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2))
            cv2.rectangle(video, p1, p2, (255, 160, 0), 2)
            cv2.circle(video, (round(track.anchor.x), round(track.anchor.y)), 4, (0, 0, 255), -1)
            cv2.putText(video, f"ID {track.track_id}", (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)

        result = cv2.copyMakeBorder(video, self.header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        signal_color = (0, 190, 0) if grant_id else (100, 100, 100)
        signal_text = f"GRANT {grant_id} ACTIVE" if grant_id else "DOOR LOCKED"
        cv2.circle(result, (25, 28), 11, signal_color, -1)
        cv2.putText(result, signal_text, (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, signal_color, 2)
        if alert != "none":
            cv2.rectangle(result, (330, 7), (790, 48), (0, 0, 210), -1)
            cv2.putText(
                result,
                alert.upper().replace("_", " "),
                (345, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
            )
        cv2.putText(
            result,
            f"t={timestamp_s:07.2f}s   entries={entries}   detector={len(tracks)}   active_tracks={len(tracks)}",
            (max(360, result.shape[1] // 2), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
        )
        stage_colors = {
            "OBSERVING": (160, 160, 160),
            "ARMED": (230, 150, 40),
            "IN_BAND": (0, 180, 255),
            "CONFIRMING": (0, 210, 255),
            "CROSSED": (0, 210, 0),
            "INSIDE": (210, 190, 40),
        }
        candidates = prioritize_debug_states(debug_states)
        card_width = max(1, result.shape[1] // 3)
        for index, state in enumerate(candidates):
            x = index * card_width + 15
            color = stage_colors.get(state.stage, (200, 200, 200))
            cv2.putText(
                result,
                f"ID {state.track_id}  {state.zone}  {state.stage}",
                (x, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
            cv2.putText(
                result,
                f"stable out={state.outside_seen}/{state.stable_frames} in={state.inside_seen}/{state.stable_frames}",
                (x, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (190, 190, 190),
                1,
            )
        if not candidates:
            cv2.putText(result, "No active candidates", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (130, 130, 130), 2)
        return result

    def write(
        self, frame: Any, tracks: Sequence[Track], timestamp_s: float,
        grant_id: str | None = None, entries: int = 0, alert: str = "none",
        debug_states: Sequence[CrossingDebugState] = (),
    ) -> None:
        self.writer.write(self.draw(frame, tracks, timestamp_s, grant_id, entries, alert, debug_states))

    def close(self) -> None:
        self.writer.release()
