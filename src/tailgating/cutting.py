from __future__ import annotations

import copy
import math
from typing import Any


def plan_event_cut(event: dict[str, Any], video_duration_s: float, padding_s: float = 5.0) -> tuple[float, float]:
    if event.get("t_end_s") is None:
        raise ValueError(f"event {event.get('event_id')} is not closed")
    return max(0.0, event["t_start_s"] - padding_s), min(video_duration_s, event["t_end_s"] + padding_s)


def rebase_annotations(
    grants: list[dict[str, Any]],
    crossings: list[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_grants = []
    for grant in grants:
        if grant["t_relock_s"] >= start_s and grant["t_unlock_s"] <= end_s:
            item = copy.deepcopy(grant)
            item["t_unlock_s"] = round(max(start_s, item["t_unlock_s"]) - start_s, 3)
            item["t_relock_s"] = round(min(end_s, item["t_relock_s"]) - start_s, 3)
            selected_grants.append(item)
    selected_crossings = []
    for crossing in crossings:
        if start_s <= crossing["t_cross_s"] <= end_s:
            item = copy.deepcopy(crossing)
            item["t_cross_s"] = round(item["t_cross_s"] - start_s, 3)
            selected_crossings.append(item)
    return selected_grants, selected_crossings


def quantize_cut_to_frames(
    start_s: float,
    end_s: float,
    fps: float,
    frame_count: int,
) -> tuple[int, int, float, float]:
    start_frame = max(0, min(frame_count, math.floor(start_s * fps)))
    end_frame = max(start_frame, min(frame_count, math.ceil(end_s * fps)))
    return start_frame, end_frame, start_frame / fps, end_frame / fps
