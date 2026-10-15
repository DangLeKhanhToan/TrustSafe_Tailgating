from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class AnnotationDocument:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._undo: list[dict[str, Any]] = []

    @classmethod
    def new(
        cls,
        session_id: str,
        video_path: str,
        camera_id: str,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        pre_grace_s: float = 0.0,
        post_grace_s: float = 0.0,
    ) -> "AnnotationDocument":
        return cls(
            {
                "schema_version": 1,
                "session_id": session_id,
                "video_path": video_path,
                "camera_id": camera_id,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "grant_pre_grace_s": pre_grace_s,
                "grant_post_grace_s": post_grace_s,
                "grants": [],
                "crossings": [],
                "events": [],
            }
        )

    @classmethod
    def load(cls, path: Path) -> "AnnotationDocument":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _remember(self) -> None:
        self._undo.append(copy.deepcopy(self.data))

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.data = self._undo.pop()
        return True

    @staticmethod
    def _next_id(items: list[dict[str, Any]], key: str, prefix: str) -> str:
        maximum = max((int(item[key][1:]) for item in items), default=0)
        return f"{prefix}{maximum + 1:03d}"

    def open_grant(self) -> dict[str, Any] | None:
        return next((grant for grant in reversed(self.data["grants"]) if grant["t_relock_s"] is None), None)

    def start_grant(self, timestamp_s: float) -> dict[str, Any]:
        if self.open_grant() is not None:
            raise ValueError("close the current grant before starting another")
        self._remember()
        grant = {
            "grant_id": self._next_id(self.data["grants"], "grant_id", "g"),
            "t_unlock_s": round(timestamp_s, 3),
            "t_relock_s": None,
        }
        self.data["grants"].append(grant)
        return grant

    def close_grant(self, timestamp_s: float) -> dict[str, Any]:
        grant = self.open_grant()
        if grant is None:
            raise ValueError("there is no open grant")
        if timestamp_s < grant["t_unlock_s"]:
            raise ValueError("grant cannot close before it starts")
        self._remember()
        grant["t_relock_s"] = round(timestamp_s, 3)
        return grant

    def active_grant(self, timestamp_s: float) -> dict[str, Any] | None:
        pre_grace_s = float(self.data.get("grant_pre_grace_s", 0.0))
        post_grace_s = float(self.data.get("grant_post_grace_s", 0.0))
        matches = [
            grant
            for grant in self.data["grants"]
            if grant["t_unlock_s"] - pre_grace_s <= timestamp_s
            and (grant["t_relock_s"] is None or timestamp_s <= grant["t_relock_s"] + post_grace_s)
        ]
        return matches[0] if len(matches) == 1 else None

    def add_crossing(self, timestamp_s: float, direction: str, person_id: str) -> dict[str, Any]:
        if direction not in {"in", "out"}:
            raise ValueError("direction must be in or out")
        self._remember()
        grant = self.active_grant(timestamp_s) if direction == "in" else None
        crossing = {
            "crossing_id": self._next_id(self.data["crossings"], "crossing_id", "c"),
            "person_id": person_id,
            "t_cross_s": round(timestamp_s, 3),
            "direction": direction,
            "grant_id": grant["grant_id"] if grant else None,
        }
        self.data["crossings"].append(crossing)
        return crossing

    def open_event(self) -> dict[str, Any] | None:
        return next((event for event in reversed(self.data["events"]) if event["t_end_s"] is None), None)

    def start_event(self, timestamp_s: float, scenario_id: int) -> dict[str, Any]:
        if self.open_event() is not None:
            raise ValueError("close the current event before starting another")
        self._remember()
        event = {
            "event_id": self._next_id(self.data["events"], "event_id", "e"),
            "scenario_id": scenario_id,
            "t_start_s": round(timestamp_s, 3),
            "t_end_s": None,
        }
        self.data["events"].append(event)
        return event

    def close_event(self, timestamp_s: float) -> dict[str, Any]:
        event = self.open_event()
        if event is None:
            raise ValueError("there is no open event")
        if timestamp_s < event["t_start_s"]:
            raise ValueError("event cannot close before it starts")
        self._remember()
        event["t_end_s"] = round(timestamp_s, 3)
        return event

    def derived_alert(self, grant: dict[str, Any]) -> str:
        people = {
            crossing["person_id"]
            for crossing in self.data["crossings"]
            if crossing["direction"] == "in" and crossing["grant_id"] == grant["grant_id"]
        }
        return "tailgating" if len(people) >= 2 else "none"

    @staticmethod
    def crossing_alert(crossing: dict[str, Any]) -> str:
        if crossing["direction"] == "in" and crossing["grant_id"] is None:
            return "unauthorised_entry"
        return "none"

    def validate(self) -> None:
        if self.open_grant() is not None:
            raise ValueError("cannot save while a grant is open; press R first")
        if self.open_event() is not None:
            raise ValueError("cannot save while an event is open; press E first")

    def validate_source(
        self,
        video_path: str,
        camera_id: str,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
    ) -> None:
        mismatches = []
        if Path(self.data["video_path"]).resolve() != Path(video_path).resolve():
            mismatches.append("video_path")
        expected = {
            "camera_id": camera_id,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }
        mismatches.extend(key for key, value in expected.items() if self.data.get(key) != value)
        if abs(float(self.data.get("fps", 0.0)) - fps) > 1e-3:
            mismatches.append("fps")
        if mismatches:
            raise ValueError("annotation source mismatch: " + ", ".join(mismatches))
