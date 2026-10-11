# Session Labeling and Event-Safe Cutting Design

## 1. Goal

Add an offline annotation workflow for long recording sessions. A user labels
grants, people, crossings, scenarios, and event ranges once on the source
video. A separate cutter converts that canonical session annotation into short,
reproducible benchmark clips with rebased timestamps and the exact grant/ground-
truth CSV formats consumed by the tailgating pipeline.

One fixed camera may contain one or more independently configured door
thresholds. Every grant, crossing, and benchmark event is scoped to a
`door_id`; a scenario describes behavior at a door and does not define camera
geometry.

The workflow supports up to three simultaneously visible actors and three
derived alert outcomes: `none`, `tailgating`, and `unauthorised_entry`. Alert
outcomes are derived from primitive labels rather than entered by hand.

## 2. Scope

### Included

- `label_session.py`: OpenCV video annotation application.
- `cut_labeled_session.py`: event-centered clip exporter.
- Canonical JSON session annotation with versioning.
- Grant IDs, person IDs, crossing IDs, scenario IDs, event ranges, notes, and
  review flags.
- Multiple `door_id` values within one camera view and recording session.
- Automatic grant suggestion for each entry crossing.
- Undo, save, resume, keyboard seeking, and annotation validation.
- Event-safe cutting with timestamp rebasing.
- Generated per-clip grant CSV, ground-truth CSV, and metadata JSON.
- Session-aware split metadata to prevent adjacent clips leaking across dataset
  splits.
- Unit and integration tests that do not require a display.
- A detailed three-session recording script for up to three actors.

### Excluded

- Bounding-box annotation.
- Frame-by-frame track annotation.
- Model-assisted pre-labeling in the first version.
- Browser or PyQt UI.
- Audio-based automatic clap detection.
- Automatic train/validation/test optimization across clips from one session.

## 3. Source-of-Truth Annotation Schema

The canonical annotation is JSON. CSV files are generated artifacts and must
not be edited independently.

```json
{
  "schema_version": 1,
  "session_id": "session_01",
  "video_path": "recordings/session_01.mp4",
  "camera_id": "door_01",
  "split": "development",
  "fps": 30.0,
  "frame_count": 18000,
  "width": 1920,
  "height": 1080,
  "actors": [
    {"person_id": "p001", "name": "actor_a", "clothing": "light shirt"}
  ],
  "grants": [
    {
      "grant_id": "g001",
      "door_id": "door_01",
      "t_unlock_s": 15.2,
      "t_relock_s": 20.4,
      "scenario_id": 1,
      "notes": "normal entry"
    }
  ],
  "crossings": [
    {
      "crossing_id": "c001",
      "person_id": "p001",
      "t_cross_s": 16.3,
      "direction": "in",
      "door_id": "door_01",
      "grant_id": "g001",
      "needs_review": false
    }
  ],
  "events": [
    {
      "event_id": "e001",
      "door_id": "door_01",
      "scenario_id": 1,
      "t_start_s": 12.0,
      "t_end_s": 23.0,
      "primary_grant_id": "g001",
      "primary_crossing_id": null,
      "notes": "normal entry"
    }
  ]
}
```

All identifiers are unique within one session. A globally unique reference is
the pair `(session_id, local_id)`.

Each grant also contains its door scope:

```json
{
  "grant_id": "g001",
  "door_id": "door_01",
  "t_unlock_s": 15.2,
  "t_relock_s": 20.4,
  "scenario_id": 1
}
```

### Camera configuration

`--camera-config` points to geometric calibration, not event labels. One config
belongs to one fixed camera and can contain multiple doors:

```yaml
camera:
  id: camera_lobby_01
  reference_width: 1920
  reference_height: 1080

doors:
  - id: door_01
    threshold: [[620, 120], [620, 900]]
    inside_reference_point: [850, 500]
    hysteresis_px: 12
    roi: [[420, 80], [900, 80], [900, 980], [420, 980]]

  - id: door_02
    threshold: [[1320, 120], [1320, 900]]
    inside_reference_point: [1550, 500]
    hysteresis_px: 12
    roi: [[1100, 80], [1750, 80], [1750, 980], [1100, 980]]
```

The configuration tool allocates `door_01`, `door_02`, and so on, renders each
door with a distinct color and label, and permits selecting or editing one door
at a time. Door ROIs may not overlap by default; unusual overlapping layouts
require an explicit override because one track could otherwise be ambiguous.

If video resolution differs from the reference resolution, coordinates are
scaled only when aspect ratio matches within tolerance. A materially different
aspect ratio is rejected and requires camera reconfiguration.

## 4. Identifier Semantics

- `grant_id` identifies one access-control authorization. The labeler allocates
  `g001`, `g002`, and so on when a grant starts.
- `person_id` identifies one physical actor within a session. Creating a new
  actor allocates `p001`, `p002`, and so on. The current actor can be reselected
  so `in -> out -> in` retains one person ID.
- `crossing_id` identifies one physical threshold crossing and auto-increments
  as `c001`, `c002`, and so on.
- `event_id` identifies one benchmark episode and auto-increments as `e001`,
  `e002`, and so on.
- `door_id` identifies one geometric threshold in the configured camera. Door
  IDs remain stable across sessions recorded with that unchanged camera.
- `scenario_id` is the brief's scenario category, not an alert class.

No identifier implies equality across namespaces. In particular, `door_01`,
`g001`, `p001`, and predicted `track_id=1` are unrelated.

## 5. Derived Ground Truth

For a grant, expected entry count is the number of unique `person_id` values
whose `direction` is `in`, whose `door_id` matches, and whose `grant_id`
matches:

```text
gt_count(grant) = |{person_id : crossing.direction = in
                                and crossing.door_id = grant.door_id
                                and crossing.grant_id = grant.grant_id}|
```

Derived grant alert:

- count 0 or 1: `none`;
- count 2 or greater: `tailgating`.

Each `in` crossing with a null `grant_id` creates an
`unauthorised_entry` ground-truth event. Exit crossings never increment grant
entry counts.

This unique-person definition ensures `p001 in -> out -> in` during one grant
counts once.

## 6. Label Application State

The application maintains these independent states:

### Playback

- current frame and timestamp;
- playing or paused;
- seek step: one frame, one second, or five seconds.

### Selection

- current person;
- current door;
- current scenario;
- selected annotation for deletion or review;
- open grant, if any;
- open event range, if any.

### Document

- clean or modified;
- last saved revision;
- undo stack;
- validation errors and warnings.

The application never allows two open grants created by the normal labeling
flow. Imported overlapping grants are preserved but highlighted as warnings.

## 7. Keyboard Interaction

### Playback

- `Space`: play/pause.
- Left/Right: seek one frame.
- Shift+Left/Shift+Right: seek one second.
- `J`/`L`: seek five seconds.
- Home/End: first/last frame.

### Grants and events

- `D`: cycle the selected configured door.
- `G`: start a new grant and allocate the next grant ID.
- `R`: close the current grant.
- `B`: begin an event range.
- `E`: end the current event range and allocate the next event ID.
- `0`–`9`: select a scenario; `0` maps to scenario 10.

### People and crossings

- `N`: create and select the next person.
- `1`–`3`: select the corresponding person when focus is not in the scenario
  selector.
- `I`: add an `in` crossing at the current frame.
- `O`: add an `out` crossing at the current frame.
- `X`: toggle `needs_review` for the selected/latest crossing.

### Editing

- `U` or Ctrl+Z: undo the last mutation.
- Delete: delete the selected/latest annotation after a short on-screen
  confirmation state.
- Ctrl+S: validate and save.
- Escape: quit; if modified, require explicit save/discard/cancel input.

Because digit keys serve both scenarios and person quick-select, the first
implementation uses `S` followed by a digit for scenarios and plain `1`–`3`
for people. `S`, then `0`, selects scenario 10.

## 8. Automatic Grant Association During Labeling

When adding an `in` crossing:

- exactly one grant active for the selected `door_id` at `t_cross_s`: auto-fill
  that `grant_id`;
- no active grant: leave `grant_id` null;
- multiple imported grants active for the selected door: pause and show
  candidates; use bracket keys to select one or mark it for review.

An `out` crossing defaults to null grant ID. The labeler displays, but does not
silently apply, configurable inference grace periods; canonical ground truth is
based on the annotator's explicit association and physical event intent.

## 9. Validation

Saving is rejected for:

- invalid video metadata;
- duplicate IDs within a namespace;
- a relock before its unlock;
- crossing/event timestamps outside video duration;
- invalid direction or scenario ID;
- a crossing referencing an unknown person or grant;
- an unknown `door_id`, or a grant/crossing/event whose door scopes disagree;
- an event whose end precedes its start;
- an open grant or open event range.

Saving is allowed with warnings for:

- overlapping grants;
- overlapping door ROIs without an explicit camera-config override;
- crossing within 0.25 seconds of a grant boundary;
- `in` crossing without a grant;
- event range without any grant or crossing;
- grant without an event range;
- annotation marked `needs_review`;
- event too close to source-video boundaries for configured padding.

## 10. Event-Safe Cutting

The cutter creates one clip per annotated event. It never cuts from arbitrary
fixed minute boundaries.

Initial desired interval:

```text
desired_start = event.t_start_s - pre_padding_s
desired_end = event.t_end_s + post_padding_s
```

Defaults:

```text
pre_padding_s = 5
post_padding_s = 5
min_clip_duration_s = 15
max_clip_duration_s = 60
```

The interval is expanded symmetrically to meet minimum duration and clipped to
the source duration. If it exceeds maximum duration, the cutter rejects the
event instead of silently dropping annotations.

All grants whose effective physical intervals intersect the event and all
crossings inside the final clip interval are included. A grant must be fully
contained after padding; otherwise the cutter expands the clip or rejects the
cut when maximum duration would be exceeded.

Timestamps are rebased:

```text
clip_time = session_time - clip_start_time
```

Each output includes provenance:

```json
{
  "source_session_id": "session_01",
  "source_video": "recordings/session_01.mp4",
  "source_start_s": 10.0,
  "source_end_s": 25.0,
  "event_id": "e001",
  "scenario_id": 1,
  "split": "development"
}
```

## 11. Output Layout

```text
recordings/
  session_01.mp4
annotations/
  session_01.json
clips/
  session_01_e001.mp4
  session_01_e001_grants.csv
  session_01_e001_gt.csv
  session_01_e001_metadata.json
```

Generated grant CSV follows the brief:

```csv
grant_id,door_id,t_unlock_s,t_relock_s
g001,door_01,5.20,10.40
```

Generated ground truth adds `crossing_id` and `door_id` while preserving the
brief's required fields:

```csv
crossing_id,person_id,t_cross_s,direction,door_id,grant_id
c001,p001,6.30,in,door_01,g001
```

The inference loader accepts the original single-door brief schema and the
extended multi-door schema. With one configured door, a missing `door_id`
implies that door. With two or more configured doors, `door_id` is mandatory.

## 12. Split and Leakage Policy

All clips derived from one source session must use the same split. The cutter
copies the session split into every clip metadata file. Validation rejects a
dataset index that assigns one `session_id` to multiple splits.

Recommended split:

- session 01: development;
- session 02: validation;
- session 03: final test.

The final-test session is not used to tune detector confidence, tracker
thresholds, hysteresis, displacement, or grace period.

## 13. Recording Protocol

Three sessions use the same fixed camera and threshold, with approximately 20
events each. Actors wait 5–8 seconds between events and return fully to their
starting sides. The operator announces session/event/scenario aloud before each
take for recovery, though audio is not required in generated clips.

### Session 01: development, normal conditions

- Clothing: actor A light, B dark, C distinct color.
- Six normal entries.
- Four close followers.
- Three side-by-side/overlap entries.
- One three-person entry.
- Two simultaneous in/out events.
- Two walk-past events.
- One unused grant.
- One unauthorized entry.

### Session 02: validation, changed clothing/behavior

- Swap light/dark clothing; add jacket or hoodie.
- Four normal entries.
- Four close followers spanning 0.2–1.5 second gaps.
- Two side-by-side/overlap entries.
- One three-person entry.
- Three hesitation/back events.
- Three large-object events using backpack, box, or cart substitute.
- One `in -> out -> in` event by the same person during one grant.
- One unauthorized entry.
- One near-threshold walk-past.

### Session 03: final test, poor conditions

- Low or back light; changed clothing; backpack; umbrella or box.
- Three normal entries.
- Three close followers.
- Two side-by-side entries.
- One three-person entry.
- Two simultaneous in/out events.
- Two hesitation events.
- Two large-object events.
- Two unauthorized entries.
- Three mixed poor-condition events.

## 14. Metrics Enabled by the Schema

- Per-grant exact unique-person count accuracy and count MAE.
- Tailgating precision, recall, and F1.
- Unauthorized-entry precision, recall, F1, and timestamp error.
- Crossing precision, recall, F1, direction accuracy, and timestamp error.
- Grant-association accuracy.
- Per-door counts, alerts, crossing metrics, and macro averages across doors.
- Repeated-crossing deduplication error rate.
- Crossing-level track-fragmentation diagnosis using ground-truth person IDs.
- All applicable metrics grouped by scenario ID and session split.

Full MOTA/HOTA are excluded because the schema does not contain frame-level
bounding boxes and identities.

## 15. Testing Strategy

- Pure schema tests cover IDs, unique-person derivation, validation, undo, and
  automatic grant suggestion.
- Multi-door tests cover door selection, per-door grants, wrong-door
  references, independent simultaneous events, and ROI ambiguity.
- Pure cutting-plan tests cover padding, minimum/maximum duration, source-edge
  clipping, grant containment, and timestamp rebasing.
- Integration tests create a tiny synthetic video when OpenCV is available,
  export clips, reopen them, and validate generated CSV/metadata.
- UI drawing and keyboard dispatch are separated from mutation commands so
  document behavior is testable without opening a window.

## 16. Success Criteria

- A saved annotation can be closed and resumed without losing IDs or selection-
  independent data.
- `in -> out -> in` for one person and grant produces a unique-person count of
  one.
- Simultaneous activity at two doors is counted independently and never shares
  grants across doors.
- Auto-generated IDs remain stable across undo/save/resume.
- Every generated clip contains its complete primary event and all referenced
  grants/crossings.
- Generated timestamps are within clip duration and match source timestamps
  after rebasing.
- All clips from one session retain one dataset split.
- The original long recording and canonical annotation remain unchanged by
  cutting.
