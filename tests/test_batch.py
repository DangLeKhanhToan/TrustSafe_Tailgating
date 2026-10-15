import json

from tailgating.batch import discover_jobs, summarize_rows


def test_discover_jobs_recursively_uses_session_config(tmp_path):
    clips = tmp_path / "clips" / "session_a"
    configs = tmp_path / "configs"
    clips.mkdir(parents=True)
    configs.mkdir()
    for suffix in (".mp4", "_grants.csv", "_gt.csv", "_metadata.json"):
        (clips / f"session_a_e001{suffix}").write_text(
            json.dumps({"scenario_id": 3}) if suffix.endswith("json") else "x",
            encoding="utf-8",
        )
    (configs / "session_a.yaml").write_text("camera: {}", encoding="utf-8")

    jobs, errors = discover_jobs(tmp_path / "clips", configs)

    assert errors == []
    assert len(jobs) == 1
    assert jobs[0].scenario_id == 3
    assert jobs[0].config.name == "session_a.yaml"


def test_nested_folder_uses_metadata_session_not_immediate_parent(tmp_path):
    clips = tmp_path / "clips" / "session_a" / "scenario_3"
    configs = tmp_path / "configs"
    clips.mkdir(parents=True)
    configs.mkdir()
    for suffix in (".mp4", "_grants.csv", "_gt.csv"):
        (clips / f"clip_01{suffix}").write_text("x", encoding="utf-8")
    (clips / "clip_01_metadata.json").write_text(
        json.dumps({"scenario_id": 3, "source_session_id": "session_a"}), encoding="utf-8"
    )
    (configs / "session_a.yaml").write_text("camera: {}", encoding="utf-8")

    jobs, errors = discover_jobs(tmp_path / "clips", configs)

    assert errors == []
    assert jobs[0].session == "session_a"


def test_summary_weights_confusion_and_frames():
    rows = [
        {"status": "ok", "frames": 100, "elapsed_s": 10.0, "tp": 1, "fp": 0, "fn": 0, "tn": 1, "exact": 2, "grants": 2, "count_mae": 0.0, "unauthorized_gt": 1, "unauthorized_pred": 1, "unauthorized_tp": 1, "unauthorized_fp": 0, "unauthorized_fn": 0},
        {"status": "ok", "frames": 50, "elapsed_s": 10.0, "tp": 0, "fp": 1, "fn": 1, "tn": 0, "exact": 0, "grants": 2, "count_mae": 1.0, "unauthorized_gt": 1, "unauthorized_pred": 0, "unauthorized_tp": 0, "unauthorized_fp": 0, "unauthorized_fn": 1},
    ]

    result = summarize_rows(rows)

    assert result["processing_fps"] == 7.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["count_accuracy"] == 0.5
    assert result["count_mae"] == 0.5
    assert result["unauthorized"]["precision"] == 1.0
    assert result["unauthorized"]["recall"] == 0.5
