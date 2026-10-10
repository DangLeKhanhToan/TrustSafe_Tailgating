# Tailgating Detection Baseline Design

## 1. Goal

Build an offline Python 3.10+ baseline that processes one fixed-camera video,
an access-grant CSV file, and one camera configuration. The system detects and
tracks people, confirms entry-direction threshold crossings, associates each
crossing with an access grant, and writes the required JSON, annotated video,
and optional evaluation metrics.

The first dataset will use a horizontal or slightly elevated side view. Entry
motion is primarily left-to-right and feet may be outside the image. The code
must not hard-code this direction: each camera defines its own threshold and
inside side.

## 2. Scope

### Included

- Required one-command inference CLI.
- OpenCV camera-configuration tool.
- YAML camera configuration.
- Pluggable person detector and tracker interfaces.
- A YOLOX-Nano ONNX detector baseline and a small motion/IoU tracker with no
  cloud inference.
- Hysteresis-based crossing state machine.
- Grant association and per-grant deduplication.
- Required JSON and annotated-video outputs.
- Evaluation against the supplied grant and ground-truth CSV schemas.
- Dataset validation and benchmark scripts.
- Unit tests for geometry, crossing, grant association, deduplication, and
  metrics.
- Documentation of assumptions, dataset protocol, dependencies, and licenses.

### Excluded from the first baseline

- Door-open visual-state detection.
- Face recognition or identity matching between tracks and credentials.
- Training or fine-tuning a detector.
- Cloud inference.
- Multi-camera identity association.
- PyQt or browser-based UI.
- Calibrated tailgating confidence.

## 3. Meaning of an Access Grant

An access grant is one authorization to unlock the door, produced in this
exercise by a row in `<clip>_grants.csv`:

```csv
grant_id,t_unlock_s,t_relock_s
g01,12.40,17.40
```

One grant authorizes one entry. The resulting classifications are:

- zero confirmed entry crossings: unused grant, `alert = "none"`;
- one confirmed entry crossing: valid use, `alert = "none"`;
- two or more confirmed entry crossings: `alert = "tailgating"`;
- an entry crossing outside every effective grant window:
  `alert = "unauthorised_entry"`;
- an exit-direction crossing never increments an entry count.

The system does not claim that the tracked person is the credential holder. It
only compares the number of observed entries with the number authorized by the
grant.

## 4. Inputs and Outputs

### Inference command

```bash
python detect_tailgating.py \
  --video clips/door_01.mp4 \
  --grants clips/door_01_grants.csv \
  --config configs/door_01.yaml \
  --out results/door_01
```

Passing `--gt clips/door_01_gt.csv` additionally writes `metrics.json`.

### Configuration command

```bash
python configure_camera.py \
  --video clips/door_01.mp4 \
  --out configs/door_01.yaml
```

### Outputs

- `events.json`: one row for every grant, including unused grants, followed by
  one row for every unauthorised entry crossing.
- `annotated.mp4`: bounding boxes, track IDs, threshold, zones, active grant,
  count, timestamp, and alert banner.
- `metrics.json`: written only when ground truth is supplied.

## 5. Camera Configuration UI

`configure_camera.py` displays a video frame with OpenCV and supports:

1. Click the two threshold endpoints.
2. Click one reference point on the inside side.
3. Render translucent `OUTSIDE` and `INSIDE` half-plane overlays.
4. Render the threshold, two hysteresis boundaries, and an entry-direction
   arrow.
5. Allow undo, reset, direction flip, frame navigation, save, and quit.
6. Save pixel coordinates in the original video coordinate system.

The tool validates that the threshold endpoints differ, the inside reference
does not lie inside the hysteresis band, and all coordinates are within the
frame.

## 6. Component Boundaries

```text
detect_tailgating.py
  -> CLI/config validation
  -> VideoReader
  -> PersonDetector
  -> MultiObjectTracker
  -> CrossingDetector
  -> GrantAssociator
  -> EventAggregator
  -> Renderer + JSON writer + Metrics
```

- `schemas.py`: typed records and validation for grants, detections, tracks,
  crossings, events, and camera configuration.
- `video.py`: deterministic frame iteration and timestamp calculation from the
  frame index and FPS.
- `detector.py`: detector protocol and a YOLOX-Nano ONNX Runtime adapter that
  filters the COCO `person` class and performs local preprocessing/postprocess.
- `tracker.py`: tracker protocol and a small constant-velocity, IoU/distance
  association baseline designed to be replaced by ByteTrack after initial
  end-to-end validation.
- `geometry.py`: signed distance, side classification, clipping, polygon
  construction, and arrow direction.
- `crossing.py`: per-track state machine only; it does not know about grants.
- `grants.py`: CSV loading, effective windows, overlap policy, and association.
- `events.py`: deduplication and event aggregation.
- `rendering.py`: all visualization and video encoding.
- `metrics.py`: event-level alert metrics, exact count accuracy, scenario table,
  and speed reporting.
- `pipeline.py`: orchestration without embedding component-specific logic.

## 7. Geometry and Anchor

The threshold is an oriented line segment. The inside reference point chooses
which sign of the line equation is `INSIDE`; reversing the two line endpoints
must not silently reverse the configured meaning.

The default anchor for the first side-view camera is the bounding-box center:

```text
x = (x1 + x2) / 2
y = (y1 + y2) / 2
```

Configuration may select `bottom_center` or a custom fractional point later.
The crossing state machine consumes anchor positions rather than bounding boxes,
so the anchor policy remains replaceable.

The line is treated as infinite for side classification, while the configured
door region limits which tracks are eligible to cross. This prevents a person
crossing the mathematical line far above or below the actual doorway from being
counted.

## 8. Crossing State Machine

Signed distance to the threshold creates three zones:

```text
OUTSIDE | HYSTERESIS BAND | INSIDE
```

Per-track states are:

```text
UNKNOWN
OUTSIDE_STABLE
IN_TRANSITION
INSIDE_CANDIDATE
INSIDE_STABLE
```

An entry crossing is emitted only when:

1. the track was observed stably outside;
2. it moved through the hysteresis band toward inside;
3. it was then observed inside for `stable_frames` consecutive observations;
4. net projected displacement in the entry direction exceeds
   `min_displacement_px`;
5. its crossing point lies inside the door region;
6. it has not already emitted the same logical crossing.

The reported crossing timestamp is the interpolated time at which the anchor
crossed the center threshold, not the later confirmation time.

The reverse transition may emit an `out` crossing for diagnostics and ground
truth comparison, but it never increments a grant entry count.

## 9. Tracking and Occlusion

The initial tracker must preserve an ID through short missed detections using a
configurable `max_lost_seconds`, converted to frames using video FPS. Association
uses motion prediction and bounding-box overlap/distance. Detector and tracker
interfaces remain replaceable with ByteTrack or another implementation.

The baseline cannot guarantee correct identity through complete overlap. Such
failures must be observable in the annotated video and reported rather than
hidden by event heuristics.

The detector requires an explicitly configured local `.onnx` file and never
downloads weights during inference. Setup documentation records the exact
weight source, checksum, model license, input resolution, and COCO class map.
The program fails early with an actionable message when the weight file is
missing or incompatible.

## 10. Grant Association and Deduplication

Each grant has an effective interval:

```text
[t_unlock_s - pre_grace_s, t_relock_s + post_grace_s]
```

Initial defaults are `pre_grace_s = 0.0` and `post_grace_s = 0.5`. The final
post-grace value must be chosen on development/validation data and then frozen
before final-test evaluation.

If a crossing lies in more than one effective grant window, it is assigned to
the latest grant whose unlock timestamp is not after the crossing. Overlapping
grant windows are reported by dataset validation.

Deduplication key:

```text
(track_id, grant_id)
```

Therefore `in -> out -> in` by the same track during one grant contributes one
entry. A later grant may count that track again. An unauthorised crossing is
deduplicated using the track ID and a configurable temporal cooldown.

## 11. Required Edge Cases

Automated tests and recorded scenarios cover:

- jitter around the threshold without a crossing;
- entering the hysteresis band and stepping back;
- normal outside-to-inside entry;
- two close tracks entering during one grant;
- side-by-side or overlapping people;
- short detection loss during transition;
- ID switch after occlusion as a documented failure mode;
- `in -> out -> in` during one grant;
- simultaneous entry and exit;
- walking near the door outside the door region;
- a grant with no entry;
- entry before unlock or after relock/grace;
- crossing exactly on grant boundaries;
- overlapping grants;
- a track first appearing inside or inside the hysteresis band;
- low FPS and skipped-frame timestamp behavior;
- malformed CSV/configuration and invalid video metadata.

## 12. Evaluation

For every grant:

```text
gt_count = number of GT rows with direction == "in" and matching grant_id
pred_count = number of unique predicted entry crossings assigned to grant_id
gt_tailgating = gt_count > 1
pred_tailgating = pred_count > 1
```

Metrics include:

- tailgating precision, recall, and F1 per grant event;
- exact count accuracy;
- count mean absolute error as an additional diagnostic;
- confusion counts grouped by scenario number;
- unauthorised-entry precision/recall with a configurable timestamp matching
  tolerance;
- end-to-end FPS and hardware/configuration metadata.

Thresholds and grace periods are selected on validation data, never on the final
test split.

## 13. Error Handling

The CLI fails with actionable messages for missing files, invalid YAML/CSV,
invalid FPS, an unwritable output directory, inconsistent timestamps, or a
model that cannot be loaded. Recoverable per-frame detection failures are logged
with frame index and timestamp. Partial success must not be presented as a
complete result.

## 14. Dataset Protocol

The primary dataset is recorded using the configured fixed camera. Entry is
mostly left-to-right, but negative examples include right-to-left exits,
near-door motion, hesitation, and simultaneous opposite directions.

Recording sessions, actor combinations, lighting, camera height, clothing, and
props are recorded in `dataset_metadata/clips.csv`. Development, validation,
and final test are split by recording session or actor combination, never by
neighboring frames. Scenario order is mixed within sessions to avoid confounding
scenario type with lighting or recording drift.

## 15. Repository Layout

The repository follows the user-provided layout. `results/` contains generated
artifacts and is not used as source input. Tests use small synthetic tracks and
CSV fixtures so core rules can be verified without a large video or model.

## 16. Success Criteria

- The required CLI runs entirely offline after dependencies and weights are
  installed.
- Output schemas match the brief exactly.
- Core temporal rules pass deterministic unit tests.
- The camera tool clearly shows inside/outside zones and direction.
- At least three recorded clips produce annotated videos, including one correct
  tailgating alert and one documented failure.
- Reported metrics and FPS are produced by executable scripts, not entered by
  hand.
