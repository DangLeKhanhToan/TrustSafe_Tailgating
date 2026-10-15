import json

from tailgating.annotation import AnnotationDocument


def test_add_grant_and_associate_entry(tmp_path):
    doc = AnnotationDocument.new(
        session_id="pilot_01",
        video_path="recordings/pilot_01.mp4",
        camera_id="pilot_01",
        fps=30.0,
        frame_count=300,
        width=640,
        height=480,
    )

    grant = doc.start_grant(1.0)
    doc.close_grant(3.0)
    crossing = doc.add_crossing(2.0, "in", person_id="p001")

    assert grant["grant_id"] == "g001"
    assert crossing["grant_id"] == "g001"
    assert doc.derived_alert(grant) == "none"

    path = tmp_path / "pilot.json"
    doc.save(path)
    loaded = AnnotationDocument.load(path)
    assert loaded.data == json.loads(path.read_text(encoding="utf-8"))


def test_tailgating_counts_unique_people_and_ignores_exit():
    doc = AnnotationDocument.new("s", "v.mp4", "door", 25.0, 250, 640, 480)
    grant = doc.start_grant(1.0)
    doc.close_grant(5.0)
    doc.add_crossing(2.0, "in", "p001")
    doc.add_crossing(3.0, "out", "p001")
    doc.add_crossing(4.0, "in", "p002")

    assert doc.derived_alert(grant) == "tailgating"


def test_entry_outside_grant_is_unauthorised():
    doc = AnnotationDocument.new("s", "v.mp4", "door", 25.0, 250, 640, 480)

    crossing = doc.add_crossing(2.0, "in", "p001")

    assert crossing["grant_id"] is None
    assert doc.crossing_alert(crossing) == "unauthorised_entry"


def test_post_grace_associates_entry_after_relock():
    doc = AnnotationDocument.new("s", "v.mp4", "door", 25.0, 250, 640, 480, post_grace_s=0.5)
    grant = doc.start_grant(1.0)
    doc.close_grant(2.0)

    assert doc.add_crossing(2.4, "in", "p001")["grant_id"] == grant["grant_id"]
    assert doc.add_crossing(2.6, "in", "p002")["grant_id"] is None


def test_source_metadata_mismatch_is_rejected():
    doc = AnnotationDocument.new("s", "v.mp4", "door", 25.0, 250, 640, 480)

    try:
        doc.validate_source("other.mp4", "door", 25.0, 250, 640, 480)
    except ValueError as exc:
        assert "video_path" in str(exc)
    else:
        raise AssertionError("metadata mismatch should fail")
