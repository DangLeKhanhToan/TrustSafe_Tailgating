# Tailgating Detection — Measured Baseline Report

## 1. System and evaluation protocol

The submitted baseline runs fully offline with YOLOX-Nano ONNX at a fixed
`416 x 416` detector input. Detections are associated by the current
`CentroidTracker`; this version does **not** claim to use ByteTrack. A
per-camera threshold line, inside reference point, hysteresis band, stable-frame
requirement, and minimum displacement determine entry crossings. Crossings are
associated with a grant using its unlock/relock interval and a 0.5 s post-grace
period. The same `(track_id, grant_id)` is counted once.

The benchmark recursively processed every labeled clip under `clips/` with
rendering enabled. It used 41 clips from four recorded sessions, 9,594 total
frames, 31 grants, and three ground-truth unauthorized entries. Ground-truth
grant counts use unique `person_id` values, so one person entering twice during
one grant still counts as one authorized person.

```powershell
python scripts\run_all.py `
  --clips clips `
  --configs configs `
  --results results\submission_baseline_416 `
  --model models\yolox_nano.onnx
```

## 2. Hardware and processing speed

- OS: Windows 11 (`10.0.26200`)
- Execution provider: ONNX Runtime `CPUExecutionProvider`
- CPU identifier available to the sandbox: Intel64 Family 6 Model 154
- Python: 3.13.13
- Detector input: fixed 416 x 416
- Frame stride: 1; every frame is processed
- Annotated rendering: enabled
- Frames: 9,594
- Pipeline time: 593.702 s
- Weighted pipeline processing throughput: **16.16 FPS**
- Per-clip range: **5.59–26.18 FPS**

Weighted FPS is `total frames / total pipeline elapsed time`. Model startup for
each CLI process is outside the pipeline timer printed by `detect_tailgating.py`.

## 3. Results

| Metric | Result |
|---|---:|
| Tailgating precision | 0.600 |
| Tailgating recall | 0.750 |
| Tailgating F1 | 0.667 |
| Tailgating confusion | TP 3, FP 2, FN 1, TN 25 |
| Exact grant-count accuracy | 0.806 (25/31) |
| Grant-count MAE | 0.226 |
| Unauthorized precision | 0.500 |
| Unauthorized recall | 0.333 |
| Unauthorized F1 | 0.400 |
| Unauthorized confusion | TP 1, FP 1, FN 2 |

Detailed tables are in `results/submission_baseline_416`: `metrics_by_clip.csv`,
`metrics_by_session.csv`, `metrics_by_scenario.csv`, and `summary.json`.

### By recording session

| Session | Clips | Grants | FPS | Tailgating F1 | Count accuracy | Count MAE |
|---|---:|---:|---:|---:|---:|---:|
| Bubi_1 | 16 | 9 | 12.21 | 0.000 | 1.000 | 0.000 |
| Bubi_2 | 8 | 7 | 24.33 | 0.000 | 0.857 | 0.143 |
| Home | 16 | 14 | 23.13 | 0.750 | 0.643 | 0.429 |
| pilot_01 | 1 | 1 | 24.28 | 0.000 | 1.000 | 0.000 |

A zero F1 for a session with no positive tailgating grants means there is no
positive evidence in that session; it does not mean every negative was wrong.

## 4. Failure analysis

### One person counted twice after track fragmentation

`Bubi_2_e005`, grant `g006`: ground truth contains the same `p001` entering at
2.200 s and 16.949 s, so the unique-person count is one. The system emitted
track 1 at 1.674 s and track 2 at 16.661 s and predicted two entries. The
current tracker cannot reconnect a person after a long disappearance. A
grant-scoped appearance gallery or re-identification tracker would address it.

### Repeated motion fragmented into three tracks

`Home_e003`, grant `g004`: ground truth is one physical person (`p002`) with two
entry crossings, which must count once. The system produced tracks 3, 2, and 5
at 8.473 s, 8.966 s, and 9.449 s and predicted three people. Close threshold
motion plus centroid association fragmented the identity. ByteTrack is the next
tracker baseline to test.

### Two overlapping people merged into one track

`Home_e012`, grant `g009`: ground truth has `p002` at 1.549 s and `p001` at
1.699 s. The system produced only track 2 at 1.550 s and missed tailgating.
This is detector-level merging under near-simultaneous overlap; a tracker cannot
recover a second person that was never detected. Higher detector resolution and
overlap-focused data are appropriate follow-ups.

The baseline also correctly found unauthorized entry `Bubi_1_e003`, falsely
raised one in `Bubi_1_e009`, and missed `Home_e014` and `Home_e015`.

## 5. Dataset limitation discovered by audit

All 41 generated metadata files currently contain `scenario_id = 1`. Therefore
the scenario table contains only scenario 1, while scenarios 2–10 are marked
missing in `summary.json`. The videos may contain those behaviors, but assigning
scenario IDs by guesswork would corrupt ground truth. Metadata must be corrected
before claiming compliance with the required per-scenario confusion table or
minimum scenario counts.

This is the largest submission gap. Overall metrics remain valid for the labels
currently present, but scenario coverage is not auditable yet.

## 6. Deployment to 500 doors

A deployment would keep calibration and inference local to each edge device,
version camera configs and models, retain short alert clips rather than
continuous video where policy permits, and send event metadata to a central
review service. Rollout should be staged per camera cohort with telemetry for
FPS, detection rate, ID fragmentation, alert rate, and input drift. Threshold
changes must be evaluated on held-out sessions; clips from one recording session
must not be split across development and final test sets.
