# Tailgating Detection

Baseline offline đếm số người đi qua một cửa cho mỗi access grant.

## Thiết lập và chạy

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python configure_camera.py --video clips/door_01.mp4 --out configs/door_01.yaml
python detect_tailgating.py --video clips/door_01.mp4 --grants clips/door_01_grants.csv --config configs/door_01.yaml --model models/yolox_nano.onnx --out results/door_01
python evaluate.py --events results/door_01/events.json --grants clips/door_01_grants.csv --gt clips/door_01_gt.csv --out results/door_01/metrics.json
```

Đặt YOLOX-Nano ONNX tại `models/yolox_nano.onnx`. Inference không tự tải model
và không gọi cloud API. `configs/door_01.yaml` chỉ chứa tọa độ minh họa; luôn
chạy `configure_camera.py` cho video thật.

Xem [ALGORITHM.md](ALGORITHM.md) để hiểu crossing state machine và grant logic.

## Test core không cần model/video

```bash
set PYTHONPATH=src
python -m unittest discover -s tests -v
```
