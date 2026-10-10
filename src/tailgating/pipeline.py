from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .crossing import CrossingDetector
from .detector import YoloXOnnxDetector
from .events import EventAggregator
from .grants import load_grants
from .schemas import CameraConfig, Event, Grant, ThresholdConfig, Track, load_camera_config
from .tracker import CentroidTracker
from .video import VideoReader


@dataclass(frozen=True)
class PipelineResult:
    events: tuple[Event, ...]
    frames: int
    elapsed_s: float

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s else 0.0


def process_track_frames(
    frames: Iterable[Sequence[Track]], grants: Sequence[Grant], threshold: ThresholdConfig,
    stable_frames: int, min_displacement_px: float, pre_grace_s: float, post_grace_s: float,
) -> list[Event]:
    crossing_detector = CrossingDetector(threshold, stable_frames, min_displacement_px)
    aggregator = EventAggregator(grants, pre_grace_s, post_grace_s)
    for tracks in frames:
        for track in tracks:
            crossing = crossing_detector.update(track)
            if crossing is not None:
                aggregator.add(crossing)
    return aggregator.finalize()


def run_pipeline(video_path: Path, grants_path: Path, config_path: Path, output_dir: Path, model_path: Path | None = None) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_camera_config(config_path)
    grants = load_grants(grants_path)
    detector = YoloXOnnxDetector(
        model_path or Path(config.detector_model), config.detector_input_size,
        config.detector_confidence, config.detector_nms_iou,
    )
    started = time.perf_counter()
    frames = 0
    crossing_detector = CrossingDetector(config.threshold, config.stable_frames, config.min_displacement_px)
    aggregator = EventAggregator(grants, config.pre_grace_s, config.post_grace_s)
    with VideoReader(video_path) as reader:
        tracker = CentroidTracker(max(1, round(config.max_lost_seconds * reader.fps)))
        from .rendering import Renderer
        renderer = Renderer(output_dir / "annotated.mp4", reader.fps, reader.width, reader.height, config)
        try:
            for frame in reader:
                tracks = tracker.update(detector.detect(frame.image), frame.index, frame.timestamp_s)
                for track in tracks:
                    crossing = crossing_detector.update(track)
                    if crossing:
                        aggregator.add(crossing)
                grant_id, entries, alert = aggregator.status_at(frame.timestamp_s)
                renderer.write(frame.image, tracks, frame.timestamp_s, grant_id, entries, alert)
                frames += 1
        finally:
            renderer.close()
    events = tuple(aggregator.finalize())
    (output_dir / "events.json").write_text(
        json.dumps([event.to_dict() for event in events], indent=2), encoding="utf-8"
    )
    return PipelineResult(events, frames, time.perf_counter() - started)
