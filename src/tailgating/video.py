from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_s: float
    image: Any


class VideoReader:
    def __init__(self, path: Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise VideoError("opencv-python is required for video processing") from exc
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise VideoError(f"cannot open video: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.capture.release()
            raise VideoError("video FPS must be positive")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[VideoFrame]:
        index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            yield VideoFrame(index, index / self.fps, image)
            index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
