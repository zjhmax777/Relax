# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Session commit keeps only accepted response tokens in the provenance record."""

import pytest

from examples.generation_provenance_demo import _accept, _commit, _request, _session, run
from relax.utils import generation_provenance as provenance


@pytest.fixture
def published(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_GENERATION_PROVENANCE_DIR", str(tmp_path))
    cursor = provenance.PublicationCursor.open(tmp_path, "policy")
    return cursor, [cursor.note_successful_publish() for _ in range(2)]


def test_partial_resume_keeps_each_attempt(published):
    cursor, tags = published
    shard, session, forest, prompt = _session()
    request = _request(prompt.state_hash, "resume")
    _accept(shard, request, "ab", tags[0], finish="abort")
    request.abort_count += 1
    cursor.note_successful_publish()
    newer = cursor.note_successful_publish()
    _accept(shard, request, "c", newer)
    _commit(shard, session, request)
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id["resume"],
        tokenizer=shard._generation_backend.tokenizer,
    )
    segments = sample.metadata[provenance.METADATA_KEY]["segments"]
    assert sample.tokens == [112] * 4 + [97, 98, 99]
    assert [(item["token_start"], item["token_end"], item["attempt_id"]) for item in segments] == [
        (4, 6, "resume#0"),
        (6, 7, "resume#1"),
    ]


def test_cancelled_attempt_is_absent_from_the_exported_sample(published):
    _, tags = published
    shard, session, forest, prompt = _session()
    cancelled = _request(prompt.state_hash, "cancelled")
    _accept(shard, cancelled, "discard", tags[0])
    replacement = _request(prompt.state_hash, "replacement")
    _accept(shard, replacement, "ok", tags[1])
    _commit(shard, session, replacement)
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id["replacement"],
        tokenizer=shard._generation_backend.tokenizer,
    )
    assert sample.response == "ok"
    assert {item["request_id"] for item in sample.metadata[provenance.METADATA_KEY]["segments"]} == {"replacement"}
    assert "cancelled" not in session.resp_state_hash_by_request_id


def test_reused_response_keeps_the_first_version(published):
    cursor, tags = published
    shard, session, forest, prompt = _session()
    first = _request(prompt.state_hash, "original")
    _accept(shard, first, "same", tags[0])
    _commit(shard, session, first)
    later = _request(prompt.state_hash, "later")
    _accept(shard, later, "same", cursor.note_successful_publish())
    _commit(shard, session, later)
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id["later"],
        tokenizer=shard._generation_backend.tokenizer,
    )
    segment = sample.metadata[provenance.METADATA_KEY]["segments"][0]
    assert segment["request_id"] == "original"
    assert segment["raw_version"] == tags[0]


@pytest.mark.parametrize("scenario", ["complete", "missing", "stale"])
def test_demo_scenarios(scenario):
    run(scenario)
