from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import BoundingBox, Detection


class DetectorError(RuntimeError):
    pass


class YoloXOnnxDetector:
    def __init__(self, model_path: Path, input_size: int = 416, confidence: float = 0.3, nms_iou: float = 0.5) -> None:
        if not model_path.is_file():
            raise DetectorError(f"YOLOX ONNX model not found: {model_path}")
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorError("numpy, opencv-python and onnxruntime are required") from exc
        self.cv2, self.np = cv2, np
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.confidence = confidence
        self.nms_iou = nms_iou

    def detect(self, frame: Any) -> list[Detection]:
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = self.cv2.resize(frame, (int(width * scale), int(height * scale)))
        canvas = self.np.full((self.input_size, self.input_size, 3), 114, dtype=self.np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(self.np.float32)
        output = self.session.run(None, {self.input_name: blob})[0]
        return self._decode(output, scale, width, height)

    def _decode(self, output: Any, scale: float, width: int, height: int) -> list[Detection]:
        prediction = output[0]
        strides = (8, 16, 32)
        grids, expanded = [], []
        for stride in strides:
            size = self.input_size // stride
            xv, yv = self.np.meshgrid(self.np.arange(size), self.np.arange(size))
            grids.append(self.np.stack((xv, yv), axis=2).reshape(1, -1, 2))
            expanded.append(self.np.full((1, size * size, 1), stride))
        grid = self.np.concatenate(grids, axis=1)[0]
        stride = self.np.concatenate(expanded, axis=1)[0]
        prediction[:, :2] = (prediction[:, :2] + grid) * stride
        prediction[:, 2:4] = self.np.exp(prediction[:, 2:4]) * stride
        scores = prediction[:, 4] * prediction[:, 5]
        keep = scores >= self.confidence
        boxes, kept_scores = prediction[keep, :4], scores[keep]
        if len(boxes) == 0:
            return []
        xyxy = self.np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) / scale
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) / scale
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) / scale
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) / scale
        xyxy[:, [0, 2]] = self.np.clip(xyxy[:, [0, 2]], 0, width - 1)
        xyxy[:, [1, 3]] = self.np.clip(xyxy[:, [1, 3]], 0, height - 1)
        nms_boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy]
        indices = self.cv2.dnn.NMSBoxes(nms_boxes, kept_scores.tolist(), self.confidence, self.nms_iou)
        return [Detection(BoundingBox(*map(float, xyxy[int(index)])), float(kept_scores[int(index)])) for index in indices]
