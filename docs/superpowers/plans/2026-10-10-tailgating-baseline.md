# Tailgating Detection Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested offline Python baseline that converts video, access grants, and a per-camera threshold configuration into entry events, alerts, an annotated video, and evaluation metrics.

**Architecture:** Domain logic is isolated from video inference: typed schemas feed geometry and a per-track crossing state machine, confirmed crossings are associated with grants and aggregated into events, and a thin pipeline connects YOLOX ONNX inference, tracking, rendering, and metrics. Synthetic observations test temporal rules without model weights; a small generated-video integration test verifies the CLI wiring.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, ONNX Runtime, PyYAML, SciPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-tailgating-detection-design.md`

## Global Constraints

- Inference is offline and must not download weights or call cloud APIs.
- The required `detect_tailgating.py` command and output schemas remain compatible with the interview brief.
- Camera direction is config-driven; the first camera uses a side view with left-to-right entry and `bbox_center` anchors.
- Core crossing and grant logic remains independent of OpenCV and ONNX Runtime.
- Python source uses concise comments only where they explain non-obvious intent; comments do not narrate straightforward syntax.
- Generated metrics and FPS come from executable code, never hand-entered values.
- The repository records dependency, model, weight, and dataset licenses.

## Review Focus

- A track first observed inside must not be counted as an entry without outside history; Task 3 tests this explicitly.
- A crossing exactly at `t_unlock_s` or `t_relock_s + post_grace_s` must associate inclusively; Task 4 tests both boundaries.
- Reversing threshold endpoint order must preserve inside/outside meaning through the inside reference point; Task 2 tests it.
- Invalid or zero video FPS must fail before timestamps are emitted; Task 6 tests metadata validation.
- Empty ground truth and zero predicted positives must return defined zero-valued metrics instead of division errors; Task 5 tests it.

---

### Task 1: Repository Skeleton, Schemas, and Configuration

**Files:**
- Create: `requirements.txt`
- Create: `src/tailgating/__init__.py`
- Create: `src/tailgating/schemas.py`
- Create: `configs/default.yaml`
- Create: `configs/door_01.yaml`
- Create: `tests/conftest.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Point`, `BoundingBox`, `Detection`, `Track`, `Grant`, `Crossing`, `Event`, `ThresholdConfig`, `CameraConfig`, and `load_camera_config(path: Path) -> CameraConfig`.
- Validation contract: invalid box order, negative timestamps, relock before unlock, coincident threshold endpoints, invalid anchor type, and non-positive stability settings raise `ValueError` with field context.

- [ ] **Step 1: Write failing schema/configuration tests**

```python
def test_grant_rejects_relock_before_unlock():
    with pytest.raises(ValueError, match="relock"):
        Grant("g01", 5.0, 4.0)

def test_config_rejects_coincident_threshold_points(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text("door:\n  threshold: [[10, 20], [10, 20]]\n  inside_reference_point: [30, 20]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_camera_config(path)
```

- [ ] **Step 2: Run the tests and verify missing imports/types fail**

Run: `python -m pytest tests/test_schemas.py -v`

Expected: collection fails because `tailgating.schemas` does not exist.

- [ ] **Step 3: Implement frozen dataclasses and YAML loading**

Use explicit parsing instead of silently accepting unknown shapes. `CameraConfig` contains detector, tracker, grant, rendering, and crossing settings with defaults applied by `load_camera_config`.

- [ ] **Step 4: Add YAML fixtures with a vertical threshold and right-side inside reference**

`configs/default.yaml` contains portable defaults; `door_01.yaml` overrides camera ID and example pixel coordinates, clearly marked as requiring regeneration for the recorded clip.

- [ ] **Step 5: Run schema tests and the full suite**

Run: `python -m pytest tests/test_schemas.py -v && python -m pytest -v`

Expected: all collected tests pass.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src configs tests
git commit -m "feat: add validated domain schemas and configuration"
```

### Task 2: Threshold Geometry and Camera Configuration Tool

**Files:**
- Create: `src/tailgating/geometry.py`
- Create: `configure_camera.py`
- Create: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Point`, `ThresholdConfig`, `CameraConfig` from Task 1.
- Produces: `signed_distance(point, start, end) -> float`, `classify_zone(point, threshold) -> Zone`, `entry_normal(threshold) -> Point`, `crossing_time(previous_point, previous_time, point, time, threshold) -> float`, and `build_half_plane_overlays(width, height, threshold) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing geometry tests with hand-derived values**

```python
def test_vertical_threshold_classifies_reference_side_as_inside():
    threshold = ThresholdConfig((50, 0), (50, 100), (80, 50), hysteresis_px=5)
    assert classify_zone((40, 50), threshold) is Zone.OUTSIDE
    assert classify_zone((52, 50), threshold) is Zone.BAND
    assert classify_zone((70, 50), threshold) is Zone.INSIDE

def test_reversing_endpoints_preserves_reference_defined_inside():
    forward = ThresholdConfig((50, 0), (50, 100), (80, 50), 5)
    reverse = ThresholdConfig((50, 100), (50, 0), (80, 50), 5)
    assert classify_zone((70, 50), forward) is Zone.INSIDE
    assert classify_zone((70, 50), reverse) is Zone.INSIDE

def test_crossing_time_interpolates_line_intersection():
    threshold = ThresholdConfig((50, 0), (50, 100), (80, 50), 2)
    assert crossing_time((40, 50), 1.0, (60, 50), 2.0, threshold) == pytest.approx(1.5)
```

- [ ] **Step 2: Run geometry tests and observe missing implementation failure**

Run: `python -m pytest tests/test_geometry.py -v`

- [ ] **Step 3: Implement normalized signed distance, reference-side normalization, zones, interpolation, and clipped polygons**

Use one concise comment to explain why multiplying by the reference-point sign makes endpoint order irrelevant.

- [ ] **Step 4: Implement `configure_camera.py` interactions**

Mouse clicks select two threshold points and one inside reference. Keyboard controls are `Space` play/pause, `[`/`]` frame navigation, `U` undo, `R` reset, `F` flip the inside point across the line, `S` save, and `Esc` exit. Render translucent blue outside, green inside, yellow band/threshold, text labels, and an entry arrow.

- [ ] **Step 5: Add behavior tests for overlay dimensions and invalid reference placement**

Assert both masks equal the original frame dimensions, do not overlap outside the band, and a reference inside the band raises `ValueError`.

- [ ] **Step 6: Run tests and syntax-check the interactive entry point**

Run: `python -m pytest tests/test_geometry.py -v && python -m py_compile configure_camera.py src/tailgating/geometry.py`

- [ ] **Step 7: Commit**

```bash
git add configure_camera.py src/tailgating/geometry.py tests/test_geometry.py
git commit -m "feat: add threshold geometry and camera setup tool"
```

### Task 3: Tracking and Crossing State Machine

**Files:**
- Create: `src/tailgating/tracker.py`
- Create: `src/tailgating/crossing.py`
- Create: `tests/test_crossing_state_machine.py`
- Create: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `Detection`, `Track`, `Crossing`, `ThresholdConfig`, and geometry helpers.
- Produces: `CentroidTracker.update(detections: Sequence[Detection], frame_index: int, timestamp_s: float) -> list[Track]` and `CrossingDetector.update(track: Track) -> Crossing | None`.

- [ ] **Step 1: Write failing crossing tests using literal anchor sequences**

```python
def test_outside_to_inside_emits_one_entry():
    detector = crossing_detector(stable_frames=2, min_displacement_px=12)
    emitted = feed(detector, track_id=7, xs=[30, 35, 45, 52, 60, 67])
    assert [(item.track_id, item.direction) for item in emitted] == [(7, "in")]
    assert emitted[0].t_cross_s == pytest.approx(0.35)

def test_band_jitter_does_not_emit_crossing():
    detector = crossing_detector(stable_frames=2, min_displacement_px=12)
    assert feed(detector, 7, [40, 48, 52, 49, 51, 45, 40]) == []

def test_track_first_seen_inside_is_not_an_entry():
    detector = crossing_detector(stable_frames=2, min_displacement_px=12)
    assert feed(detector, 7, [65, 70, 75]) == []
```

- [ ] **Step 2: Run crossing tests and verify RED**

Run: `python -m pytest tests/test_crossing_state_machine.py -v`

- [ ] **Step 3: Implement per-track state and interpolated candidate crossing**

Store outside/inside stability counters, transition origin, maximum projected displacement, previous observation, and emitted direction history. A missed observation preserves state until tracker expiry; a newly created track inside cannot synthesize outside history.

- [ ] **Step 4: Add tests for hesitation, reverse exit, short missed observations, and repeated `in -> out -> in`**

The crossing detector may emit two `in` crossings for the raw trajectory; grant-level deduplication in Task 4 must reduce them to one entry for the same grant.

- [ ] **Step 5: Write failing tracker tests**

Test ID preservation across steady motion, one missing frame, expiration after `max_lost_frames`, and two non-overlapping detections receiving different IDs.

- [ ] **Step 6: Implement constant-velocity IoU/distance association**

Build a cost matrix from predicted-center distance and IoU, reject pairs above configured gates, solve assignment with SciPy, update matched tracks, retain temporarily lost tracks, and create IDs for unmatched detections.

- [ ] **Step 7: Run component and full tests**

Run: `python -m pytest tests/test_crossing_state_machine.py tests/test_tracker.py -v && python -m pytest -v`

- [ ] **Step 8: Commit**

```bash
git add src/tailgating/tracker.py src/tailgating/crossing.py tests
git commit -m "feat: add temporal tracking and crossing state machine"
```

### Task 4: Grant Association and Event Aggregation

**Files:**
- Create: `src/tailgating/grants.py`
- Create: `src/tailgating/events.py`
- Create: `tests/test_grant_association.py`
- Create: `tests/test_deduplication.py`
- Create: `tests/fixtures/grants.csv`

**Interfaces:**
- Consumes: `Grant`, `Crossing`, and `Event` schemas.
- Produces: `load_grants(path: Path) -> list[Grant]`, `find_grant(crossing, grants, pre_grace_s, post_grace_s) -> Grant | None`, `find_overlaps(grants, pre_grace_s, post_grace_s) -> list[tuple[str, str]]`, and `EventAggregator.add(crossing)`, `EventAggregator.finalize() -> list[Event]`.

- [ ] **Step 1: Write failing boundary and overlap tests**

```python
@pytest.mark.parametrize("timestamp", [10.0, 15.5])
def test_effective_grant_boundaries_are_inclusive(timestamp):
    grant = Grant("g01", 10.0, 15.0)
    crossing = Crossing(1, timestamp, "in")
    assert find_grant(crossing, [grant], 0.0, 0.5) == grant

def test_overlap_selects_latest_unlocked_grant():
    grants = [Grant("g01", 10, 15), Grant("g02", 14, 19)]
    assert find_grant(Crossing(1, 14.5, "in"), grants, 0, 0).grant_id == "g02"
```

- [ ] **Step 2: Run grant tests and verify RED**

Run: `python -m pytest tests/test_grant_association.py -v`

- [ ] **Step 3: Implement strict CSV parsing and deterministic association**

Reject duplicate IDs, invalid floats, unsorted/inconsistent rows, and relock-before-unlock. Sort valid grants by unlock time while retaining IDs.

- [ ] **Step 4: Write failing aggregation tests**

Assert zero-entry grants are present, one entry is `none`, two unique tracks produce `tailgating`, exits are ignored, one track entering twice in one grant counts once, the same track can count in a later grant, and an unmatched entry creates an unauthorized record.

- [ ] **Step 5: Implement `EventAggregator` with explicit deduplication sets**

Use `(track_id, grant_id)` for authorized events and `(track_id, cooldown_bucket)` behavior for unmatched entry crossings. Preserve crossings in timestamp order and grants in unlock order.

- [ ] **Step 6: Run grant, aggregation, and full tests**

Run: `python -m pytest tests/test_grant_association.py tests/test_deduplication.py -v && python -m pytest -v`

- [ ] **Step 7: Commit**

```bash
git add src/tailgating/grants.py src/tailgating/events.py tests
git commit -m "feat: associate crossings with access grants"
```

### Task 5: Metrics and Dataset Validation

**Files:**
- Create: `src/tailgating/metrics.py`
- Create: `evaluate.py`
- Create: `scripts/validate_dataset.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_dataset_validation.py`
- Create: `tests/fixtures/ground_truth.csv`

**Interfaces:**
- Consumes: predicted `Event` objects, ground-truth CSV rows, optional scenario metadata, elapsed processing seconds, processed frame count, and hardware information.
- Produces: `evaluate_events(events, grants, ground_truth, scenarios=None) -> dict`, `compute_fps(frame_count, elapsed_s) -> float`, and validation issue records with severity, file, row, and message.

- [ ] **Step 1: Write failing metric tests with hand-calculated confusion values**

Use three grants with GT counts `[1, 2, 3]` and predicted counts `[1, 1, 3]`; assert `TP=1`, `FN=1`, precision `1.0`, recall `0.5`, F1 `2/3`, and exact count accuracy `2/3`.

- [ ] **Step 2: Add empty-input tests**

Assert empty inputs and zero positive predictions return finite zeros, not `NaN` or division errors. Assert scenario grouping preserves scenarios that contain only true negatives.

- [ ] **Step 3: Run metrics tests and verify RED**

Run: `python -m pytest tests/test_metrics.py -v`

- [ ] **Step 4: Implement metrics and JSON serialization**

Use one safe division helper and convert NumPy scalar values to ordinary Python types before writing JSON.

- [ ] **Step 5: Write failing dataset-validation tests**

Exercise missing companion files, duplicate grant IDs, GT references to unknown grants, invalid directions, timestamp beyond video duration, and overlapping grants reported as warnings.

- [ ] **Step 6: Implement the evaluation CLI and validator**

`evaluate.py` reads an existing `events.json`; `validate_dataset.py` scans clip stems and returns a non-zero exit code for errors while printing warnings separately.

- [ ] **Step 7: Run tests and CLI help checks**

Run: `python -m pytest tests/test_metrics.py tests/test_dataset_validation.py -v && python evaluate.py --help && python scripts/validate_dataset.py --help`

- [ ] **Step 8: Commit**

```bash
git add src/tailgating/metrics.py evaluate.py scripts/validate_dataset.py tests
git commit -m "feat: add event evaluation and dataset validation"
```

### Task 6: Video, Detector, Rendering, and End-to-End Pipeline

**Files:**
- Create: `src/tailgating/video.py`
- Create: `src/tailgating/detector.py`
- Create: `src/tailgating/rendering.py`
- Create: `src/tailgating/pipeline.py`
- Create: `detect_tailgating.py`
- Create: `tests/test_video.py`
- Create: `tests/test_detector_postprocess.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all prior domain components plus a local YOLOX-Nano ONNX weight path.
- Produces: `VideoReader`, `YoloXOnnxDetector.detect(frame) -> list[Detection]`, `Renderer.draw(...) -> np.ndarray`, `run_pipeline(options) -> PipelineResult`, and the required CLI.

- [ ] **Step 1: Write failing video tests**

Generate a five-frame temporary video with OpenCV; assert frame indices and timestamps are `[0.0, 0.1, 0.2, 0.3, 0.4]` at 10 FPS. Patch only the capture metadata boundary to assert zero FPS raises `VideoError` before iteration.

- [ ] **Step 2: Implement deterministic video reading and output writer validation**

Timestamp every frame as `frame_index / fps`; verify capture/writer open state and retain original dimensions.

- [ ] **Step 3: Write failing YOLOX postprocess tests**

Supply literal synthetic output tensors and assert coordinate de-letterboxing, objectness-times-class score, person-class filtering, confidence filtering, clipping, and NMS results. These tests call postprocess directly and do not mock ONNX Runtime.

- [ ] **Step 4: Implement local YOLOX ONNX preprocessing and postprocess**

Load ONNX Runtime lazily, validate model input/output shapes, letterbox with recorded scale/padding, filter COCO class 0, and apply OpenCV NMS. Missing weights fail before the video loop.

- [ ] **Step 5: Write failing renderer and pipeline integration tests**

Inject a deterministic detector that returns scripted detections into `run_pipeline`; process a generated video and real grant/config fixtures. Assert `events.json` schema, zero/one/two-person decisions, annotated video readability, and `metrics.json` creation when GT is supplied.

- [ ] **Step 6: Implement rendering and pipeline orchestration**

Render boxes, IDs, anchor points, colored zones, threshold/band, direction arrow, timestamp, active grant, entry count, and alert. Keep dependency injection available for tests while the CLI constructs production components.

- [ ] **Step 7: Implement `detect_tailgating.py` argument validation and exit behavior**

Support the required arguments plus `--gt`, `--model`, and `--device`. Print output paths, processed frames, elapsed seconds, and end-to-end FPS after successful completion.

- [ ] **Step 8: Run integration and full tests**

Run: `python -m pytest tests/test_video.py tests/test_detector_postprocess.py tests/test_pipeline.py -v && python -m pytest -v && python detect_tailgating.py --help`

- [ ] **Step 9: Commit**

```bash
git add src/tailgating/video.py src/tailgating/detector.py src/tailgating/rendering.py src/tailgating/pipeline.py detect_tailgating.py tests
git commit -m "feat: add offline end-to-end inference pipeline"
```

### Task 7: Batch Scripts, Documentation, and Algorithm Explanation

**Files:**
- Create: `scripts/run_all.py`
- Create: `scripts/benchmark.py`
- Create: `README.md`
- Create: `ALGORITHM.md`
- Create: `REPORT.md`
- Create: `DATASET.md`
- Create: `ASSUMPTIONS.md`
- Create: `LICENSES.md`
- Create: `dataset_metadata/clips.csv`
- Create: `clips/.gitkeep`
- Create: `results/.gitkeep`
- Create: `.gitignore`

**Interfaces:**
- Consumes: public CLIs from Tasks 2, 5, and 6.
- Produces: reproducible user workflow, algorithm rationale, dataset recording checklist, license inventory, batch runner, and benchmark report.

- [ ] **Step 1: Implement batch and benchmark scripts against the public CLI/module interfaces**

`run_all.py` discovers `*_grants.csv`, requires matching video/config files, runs each clip in a stable order, and returns non-zero if any clip fails. `benchmark.py` records platform, Python, CPU, RAM when available, model/input size, frames, elapsed time, and end-to-end FPS as JSON.

- [ ] **Step 2: Write `ALGORITHM.md`**

Explain grant semantics, detector/tracker separation, anchor choice for cropped side views, reference-normalized signed distance, hysteresis state transitions, interpolated crossing time, grant-window association, deduplication, metrics, complexity, limitations, and expected failure modes. Include concise equations and one worked timeline example.

- [ ] **Step 3: Write setup and operation documentation**

`README.md` provides installation and execution in five commands or fewer after the model file exists. `DATASET.md` gives the scenario recording/annotation checklist. `ASSUMPTIONS.md` lists every temporal and camera assumption. `LICENSES.md` records exact project URLs/licenses and instructs the user to add the downloaded weight checksum.

- [ ] **Step 4: Add repository hygiene and empty artifact directories**

Ignore model weights, videos, generated result contents, caches, and virtual environments while retaining `.gitkeep` files and CSV metadata templates.

- [ ] **Step 5: Run all verification commands from a clean process**

Run:

```bash
python -m pytest -v
python -m compileall -q src detect_tailgating.py configure_camera.py evaluate.py scripts
python detect_tailgating.py --help
python configure_camera.py --help
python evaluate.py --help
python scripts/run_all.py --help
python scripts/benchmark.py --help
python scripts/validate_dataset.py --help
git diff --check
```

- [ ] **Step 6: Inspect source comments and remove narration**

Retain comments only for reference-sign normalization, crossing timestamp interpolation, state-machine safeguards, and non-obvious tracker association decisions.

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "docs: add reproducible workflow and algorithm guide"
```

### Task 8: Final Review and Repair

**Files:**
- Review: all files created in Tasks 1–7.
- Modify: only files implicated by review findings.

**Interfaces:**
- Consumes: complete implementation and design specification.
- Produces: reviewed baseline with critical and important findings resolved.

- [ ] **Step 1: Request an independent code review against the specification**

Provide the reviewer the spec path, plan path, base commit before implementation, final commit, and require prioritized findings covering correctness, output-schema compatibility, temporal edge cases, dependency/license risks, and test gaps.

- [ ] **Step 2: Reproduce each critical or important finding with a failing test**

Run the narrow test and confirm it fails for the reported behavior before editing production code.

- [ ] **Step 3: Apply minimal repairs and rerun narrow tests**

Keep the public interfaces stable unless the review proves the plan interface incorrect.

- [ ] **Step 4: Run the complete verification suite again**

Run the Task 7 Step 5 command block and record exact pass counts and any environment-dependent limitation, especially absence of a real YOLOX weight or user video.

- [ ] **Step 5: Commit review repairs if any**

```bash
git add .
git commit -m "fix: address baseline review findings"
```
