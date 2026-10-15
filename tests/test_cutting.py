from tailgating.cutting import plan_event_cut, quantize_cut_to_frames, rebase_annotations


def test_plan_event_cut_adds_padding_and_rebases():
    event = {"event_id": "e001", "t_start_s": 10.0, "t_end_s": 20.0}
    start, end = plan_event_cut(event, video_duration_s=60.0, padding_s=5.0)
    assert (start, end) == (5.0, 25.0)

    grants = [{"grant_id": "g001", "t_unlock_s": 12.0, "t_relock_s": 18.0}]
    crossings = [{"crossing_id": "c001", "t_cross_s": 14.0, "direction": "in"}]
    rebased_grants, rebased_crossings = rebase_annotations(grants, crossings, start, end)
    assert rebased_grants[0]["t_unlock_s"] == 7.0
    assert rebased_crossings[0]["t_cross_s"] == 9.0


def test_plan_event_cut_clamps_to_video_edges():
    event = {"event_id": "e001", "t_start_s": 1.0, "t_end_s": 4.0}
    assert plan_event_cut(event, 12.0, padding_s=5.0) == (0.0, 9.0)


def test_rebase_excludes_annotations_outside_clip():
    grants = [
        {"grant_id": "g001", "t_unlock_s": 1.0, "t_relock_s": 2.0},
        {"grant_id": "g002", "t_unlock_s": 11.0, "t_relock_s": 12.0},
    ]
    crossings = [
        {"crossing_id": "c001", "t_cross_s": 1.5, "direction": "in"},
        {"crossing_id": "c002", "t_cross_s": 11.5, "direction": "in"},
    ]

    rebased_grants, rebased_crossings = rebase_annotations(grants, crossings, 10.0, 15.0)

    assert [item["grant_id"] for item in rebased_grants] == ["g002"]
    assert [item["crossing_id"] for item in rebased_crossings] == ["c002"]


def test_quantized_cut_uses_actual_frame_boundaries():
    start_frame, end_frame, actual_start, actual_end = quantize_cut_to_frames(1.001, 2.001, 30.0, 300)

    assert (start_frame, end_frame) == (30, 61)
    assert actual_start == 1.0
    assert actual_end == 61 / 30.0
