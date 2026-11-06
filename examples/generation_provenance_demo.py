# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Show where each turn's tokens came from after two publishes during a tool wait.

The model and the weight update are stand-ins. The session commit, the loss
mask, and the published-sequence arithmetic are the real ones. Run from the
repository root:

    python -m examples.generation_provenance_demo --scenario complete
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.service import AgenticSessionShard, ResidentGroup, _SessionRecord, _SessionResultCell
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils import generation_provenance as provenance
from relax.utils.generation_provenance import PublicationCursor


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _shard():
    shard = object.__new__(AgenticSessionShard.__ray_actor_class__)
    shard.args = SimpleNamespace(
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        use_rollout_routing_replay=False,
    )
    shard._generation_backend = SimpleNamespace(tokenizer=_Tokenizer())
    return shard


def _session():
    forest = SessionForest.create_empty(session_id="session-a", index=0, group_index=0)
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "pppp"}]),
        train_token_delta=[112] * 4,
        rollout_token_delta=[112] * 4,
    )
    group = ResidentGroup(rollout_mode="train", group_id="group", result_cells={}, sessions=[])
    session = _SessionRecord(
        group=group,
        session_id="session-a",
        session_sampling_params={},
        result_cell=_SessionResultCell(),
    )
    session.forest = forest
    return _shard(), session, forest, prompt


def _request(parent, request_id):
    return InflightRequest(
        request_id=request_id,
        parent_state_hash=parent,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.new_event_loop().create_future(),
        wall_started_at=0.0,
        sampling_params={"max_new_tokens": 8},
    )


def _accept(shard, request, text, version, finish="stop"):
    meta = {"finish_reason": {"type": finish}}
    if version is not None:
        meta["weight_version"] = version
        meta["weight_versions"] = [{"start": 0, "end": len(text), "version": version}]
    shard._apply_generate_result(
        request,
        BackendGenerateResult(
            new_tokens=[ord(char) for char in text],
            new_log_probs=[-0.2] * len(text),
            finish_type=finish,
            meta_info=meta,
            elapsed=0.1,
        ),
    )


def _commit(shard, session, request):
    request.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=request, finish_type="stop")
    return session.resp_state_hash_by_request_id[request.request_id]


def run(scenario: str = "complete") -> dict:
    previous = os.environ.get("RELAX_GENERATION_PROVENANCE_DIR")
    with tempfile.TemporaryDirectory(prefix="provenance-") as directory:
        os.environ["RELAX_GENERATION_PROVENANCE_DIR"] = directory
        try:
            payload = _scenario(directory, scenario)
        finally:
            if previous is None:
                os.environ.pop("RELAX_GENERATION_PROVENANCE_DIR", None)
            else:
                os.environ["RELAX_GENERATION_PROVENANCE_DIR"] = previous
    _expect(payload)
    return payload


def _scenario(directory: str, scenario: str) -> dict:
    cursor = PublicationCursor.open(directory, "demo-policy")
    first = cursor.note_successful_publish()
    shard, session, forest, prompt = _session()
    events = []

    opening = _request(prompt.state_hash, "turn-0")
    _accept(shard, opening, "abc", first)
    _commit(shard, session, opening)
    events.append({"event": "turn_committed", "turn": 0, "published": cursor.published})

    events.append({"event": "tool_wait_begin"})
    for _ in range(2):
        cursor.note_successful_publish()
        events.append({"event": "publish_during_tool_wait", "published": cursor.published})
    events.append({"event": "tool_wait_end"})

    parent = session.resp_state_hash_by_request_id["turn-0"]
    observation = forest.append_obs(
        parent_state_hash=parent,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "tool", "tool_call_id": "tool-0", "content": "ok"}],
        train_token_delta=[111, 107],
        rollout_token_delta=[111, 107],
    )
    if scenario == "missing":
        version = None
    elif scenario == "stale":
        version = first
    else:
        version = f"pub/{cursor.run_id}/{cursor.published}"
    follow = _request(observation.state_hash, "turn-1")
    _accept(shard, follow, "de", version)
    _commit(shard, session, follow)
    events.append({"event": "turn_committed", "turn": 1, "published": cursor.published})

    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id["turn-1"],
        tokenizer=_Tokenizer(),
    )
    cursor.note_successful_publish()
    events.append({"event": "actor_loads_batch", "published": cursor.published})
    report = provenance.attribute_row(
        sample.metadata[provenance.METADATA_KEY],
        len(sample.tokens),
        sample.response_length,
        sample.loss_mask,
        provenance.read_head(),
    )
    return {"scenario": scenario, "events": events, "report": report}


def _expect(payload: dict) -> None:
    report = payload["report"]
    assert report["trained_token_count"] == 5
    assert report["attempt_version_span"] == 0
    assert report["lag_after_final_commit"] == 1
    assert report["turns"][0]["publishes_before_next_commit"] == 2
    assert report["turns"][0]["publishes_since_commit"] == 3
    assert report["turns"][1]["publishes_since_commit"] == 1
    if payload["scenario"] == "complete":
        assert report["unknown_fraction"] == 0
        assert report["weighted_version_age"] == 2.2
        assert report["trajectory_version_span"] == 2
        assert [segment["source_seq"] for segment in report["segments"]] == [1, 3]
    elif payload["scenario"] == "missing":
        assert report["unknown_fraction"] == 0.4
        assert report["weighted_version_age"] == 3
        assert report["trajectory_version_span"] == 0
    else:
        assert report["unknown_fraction"] == 0
        assert report["weighted_version_age"] == 3
        assert report["trajectory_version_span"] == 0
        assert [segment["source_seq"] for segment in report["segments"]] == [1, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("complete", "missing", "stale"), default="complete")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.scenario)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
