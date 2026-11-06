# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from relax.utils import generation_provenance as provenance


@pytest.fixture
def cursor(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_GENERATION_PROVENANCE_DIR", str(tmp_path))
    opened = provenance.PublicationCursor.open(tmp_path, "policy")
    tags = [opened.note_successful_publish() for _ in range(3)]
    return opened, tags


def _row(cursor, *, second=None, mask=None):
    run = cursor[0].run_id
    first, _, third = cursor[1]
    segments = [
        {
            "token_start": 4,
            "token_end": 7,
            "turn": 0,
            "request_id": "turn-0",
            "attempt_id": "turn-0#0",
            "piece": 0,
            "evidence": "intervals",
            "raw_version": first,
        },
        {
            "token_start": 9,
            "token_end": 11,
            "turn": 1,
            "request_id": "turn-1",
            "attempt_id": "turn-1#0",
            "piece": 0,
            "evidence": "intervals",
            "raw_version": second if second is not None else third,
        },
    ]
    if second == "drop":
        segments[1]["raw_version"] = None
        segments[1]["evidence"] = "absent"
    return {
        "session_id": "session-a",
        "turns": [
            {"turn": 0, "commit_published": 1, "commit_run": run},
            {"turn": 1, "commit_published": 3, "commit_run": run},
        ],
        "segments": segments,
    }


def test_tool_wait_is_not_the_same_number_as_queue_lag(cursor):
    cursor[0].note_successful_publish()
    report = provenance.attribute_row(_row(cursor), 11, 7, [1, 1, 1, 0, 0, 1, 1], provenance.read_head())
    assert report["trained_token_count"] == 5
    assert report["unknown_fraction"] == 0
    assert report["trajectory_version_span"] == 2
    assert report["attempt_version_span"] == 0
    assert report["weighted_version_age"] == 2.2
    assert report["oldest_age"] == 3
    assert report["newest_age"] == 1
    assert report["lag_after_final_commit"] == 1
    assert report["turns"][0]["publishes_before_next_commit"] == 2
    assert report["turns"][0]["publishes_since_commit"] == 3
    assert [segment["source_seq"] for segment in report["segments"]] == [1, 3]


def test_missing_version_stays_unknown(cursor):
    cursor[0].note_successful_publish()
    report = provenance.attribute_row(_row(cursor, second="drop"), 11, 7, [1, 1, 1, 0, 0, 1, 1], provenance.read_head())
    assert report["unknown_fraction"] == 0.4
    assert report["unknown_reasons"]["absent"] == 2
    assert report["weighted_version_age"] == 3
    assert report["trajectory_version_span"] == 0
    assert 0 not in [segment["source_seq"] for segment in report["segments"]]


@pytest.mark.parametrize("raw", ["0", "latest", None, "pub/not-a-run/1"])
def test_raw_labels_are_not_published_sequences(cursor, raw):
    source = provenance.resolve_source(raw, provenance.read_head())
    assert source["seq"] is None
    assert source["unknown"]


def test_intervals_inside_one_attempt_are_a_decode_span(cursor):
    tag_old, _, tag_new = cursor[1]
    segments = provenance.segments_from_attempt(
        {"weight_versions": [{"start": 0, "end": 2, "version": tag_old}, {"start": 2, "end": 4, "version": tag_new}]},
        4,
        offset=5,
        request_id="req",
        attempt_id="req#1",
    )
    assert [(item["start"], item["end"], item["raw_version"]) for item in segments] == [
        (5, 7, tag_old),
        (7, 9, tag_new),
    ]
    record = {
        "session_id": "s",
        "turns": [{"turn": 0, "commit_published": 3, "commit_run": cursor[0].run_id}],
        "segments": [
            {
                "token_start": item["start"],
                "token_end": item["end"],
                "turn": 0,
                "attempt_id": item["attempt_id"],
                "evidence": "intervals",
                "raw_version": item["raw_version"],
            }
            for item in segments
        ],
    }
    report = provenance.attribute_row(record, 9, 4, [1, 1, 1, 1], provenance.read_head())
    assert report["attempt_version_span"] == 2
    assert report["trajectory_version_span"] == 2


def test_broken_intervals_do_not_invent_a_version(cursor):
    segments = provenance.segments_from_attempt(
        {"weight_versions": [{"start": 0, "end": 2}, {"start": 1, "end": 3}], "weight_version": cursor[1][0]},
        3,
        offset=0,
        request_id="req",
        attempt_id="req#0",
    )
    assert len(segments) == 1
    assert segments[0]["raw_version"] is None
    assert segments[0]["evidence"] == "broken_intervals"


def test_future_sequence_is_neither_age_nor_missing(cursor):
    future = f"pub/{cursor[0].run_id}/{cursor[0].published + 5}"
    report = provenance.attribute_row(_row(cursor, second=future), 11, 7, [1, 1, 1, 0, 0, 1, 1], provenance.read_head())
    assert report["future_token_count"] == 2
    assert report["unknown_fraction"] == 0
    assert report["weighted_version_age"] == 2
    assert report["unknown_reasons"]["future_publication"] == 2


def test_overlapping_segments_are_a_bad_record(cursor):
    record = _row(cursor)
    record["segments"][1]["token_start"] = 6
    report = provenance.attribute_row(record, 11, 7, [1, 1, 1, 0, 0, 1, 1], provenance.read_head())
    assert report["record_error"] == "segment_overlap_or_out_of_range"
    assert report["unknown_fraction"] == 1
    assert report["weighted_version_age"] is None


def test_masked_and_removed_tokens_are_not_aged(cursor):
    report = provenance.attribute_row(_row(cursor), 11, 7, [0, 0, 0, 0, 0, 0, 0], provenance.read_head())
    assert report["trained_token_count"] == 0
    assert report["weighted_version_age"] is None
    assert report["unknown_fraction"] is None


def test_failed_write_does_not_skip_a_sequence(tmp_path):
    blocker = tmp_path / "occupied"
    blocker.write_text("file")
    cursor = provenance.PublicationCursor.open(blocker, "policy")
    with pytest.raises(OSError):
        cursor.note_successful_publish()
    assert cursor.published == 0
    cursor.directory = tmp_path
    assert cursor.note_successful_publish().endswith("/1")


def test_disabled_reads_nothing(monkeypatch):
    monkeypatch.delenv("RELAX_GENERATION_PROVENANCE_DIR", raising=False)
    assert provenance.enabled() is False
    assert provenance.read_head() is None


def test_round_trip_keeps_session_text():
    record = {"session_id": "会话", "turns": [], "segments": []}
    assert provenance.unpack_record(provenance.pack_record(record)) == record


def test_bad_payload_keeps_the_mask_count(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_GENERATION_PROVENANCE_DIR", str(tmp_path))
    summary = provenance.consume_loaded_batch(
        {
            provenance.RECORD_FIELD: [[255]],
            "total_lengths": [3],
            "response_lengths": [2],
            "loss_masks": [[1, 1]],
        }
    )
    assert summary["trained_token_count"] == 2
    assert summary["unknown_fraction"] == 1
    assert summary["record_errors"] == 1
    saved = json.loads(next((tmp_path / "diagnostics").glob("rank-0.summary.json")).read_text())
    assert saved["trained_token_count"] == 2
