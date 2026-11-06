# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Side record of which published policy produced each trained token.

Nothing in here changes sampling, rewards, or GRPO. The record is kept only
when ``RELAX_GENERATION_PROVENANCE_DIR`` is set, and training ignores it.

A published sequence starts at 1 for the first successful rollout weight
publish. Tokens with no confirmed publication stay unknown; they are never
rewritten as version 0 or as the newest version.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from relax.utils.env import Envs


METADATA_KEY = "generation_provenance"
RECORD_FIELD = "provenance_records"
_TAG = re.compile(r"^pub/([0-9a-f]{32})/([1-9][0-9]*)$")


def enabled() -> bool:
    return bool(Envs.RELAX_GENERATION_PROVENANCE_DIR)


def _directory() -> Path:
    return Path(Envs.RELAX_GENERATION_PROVENANCE_DIR)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class PublicationCursor:
    """One policy's successful rollout publishes, stored in the shared directory.

    ``open`` reloads the same run. A new run is created only when the file is
    absent. The counter advances after the file is stored, so a failed write
    does not skip a sequence number.
    """

    def __init__(self, directory: Path, policy: str, run_id: str, published: int) -> None:
        self.directory = directory
        self.policy = policy
        self.run_id = run_id
        self.published = published

    @property
    def _path(self) -> Path:
        return self.directory / "publication.json"

    @classmethod
    def open(cls, directory: str | Path, policy: str) -> "PublicationCursor":
        root = Path(directory)
        path = root / "publication.json"
        if not path.exists():
            return cls(root, policy, uuid.uuid4().hex, 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("policy") != policy or not isinstance(payload.get("run_id"), str):
            raise ValueError(f"publication file {path} does not belong to policy {policy}")
        published = payload.get("published")
        if type(published) is not int or published < 0:
            raise ValueError(f"publication file {path} has no usable sequence")
        return cls(root, policy, payload["run_id"], published)

    def note_successful_publish(self) -> str:
        nxt = self.published + 1
        _atomic_write(
            self._path,
            {"policy": self.policy, "run_id": self.run_id, "published": nxt},
        )
        self.published = nxt
        return f"pub/{self.run_id}/{nxt}"

    def head(self) -> dict[str, Any]:
        return {"policy": self.policy, "run_id": self.run_id, "published": self.published}


def read_head() -> dict[str, Any] | None:
    """Return the shared publication head, or None when it cannot be trusted."""

    if not enabled():
        return None
    path = _directory() / "publication.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    published = payload.get("published")
    run_id = payload.get("run_id")
    policy = payload.get("policy")
    if type(published) is not int or published < 0 or not isinstance(run_id, str) or not isinstance(policy, str):
        return None
    return {"policy": policy, "run_id": run_id, "published": published}


def resolve_source(raw: Any, head: dict[str, Any] | None) -> dict[str, Any]:
    """Map an engine version string onto a published sequence, or say why not."""

    if raw is None:
        return {"raw": None, "seq": None, "run_id": None, "unknown": "absent"}
    text = raw if isinstance(raw, str) else str(raw)
    matched = _TAG.fullmatch(text)
    if matched is None:
        return {"raw": text, "seq": None, "run_id": None, "unknown": "not_a_publication"}
    run_id, seq = matched.group(1), int(matched.group(2))
    if head is None:
        return {"raw": text, "seq": None, "run_id": run_id, "unknown": "no_publication_record"}
    if run_id != head["run_id"]:
        return {"raw": text, "seq": None, "run_id": run_id, "unknown": "outside_publication"}
    if seq > head["published"]:
        return {"raw": text, "seq": seq, "run_id": run_id, "unknown": "future_publication"}
    return {"raw": text, "seq": seq, "run_id": run_id, "unknown": None}


def segments_from_attempt(
    meta: dict[str, Any],
    token_count: int,
    *,
    offset: int,
    request_id: str,
    attempt_id: str,
) -> list[dict[str, Any]]:
    """Token ranges for one accepted attempt, relative to the response so far.

    Explicit backend intervals are used only when they tile ``[0, token_count)``
    without gaps or overlap. Otherwise the whole attempt is one range and the
    version is unknown rather than guessed.
    """

    if token_count <= 0:
        return []
    pieces = meta.get("weight_versions")
    tiled: list[tuple[int, int, Any]] | None = None
    if isinstance(pieces, list) and pieces:
        cursor = 0
        parsed: list[tuple[int, int, Any]] = []
        for piece in pieces:
            if not isinstance(piece, dict):
                parsed = []
                break
            start, end = piece.get("start"), piece.get("end")
            if type(start) is not int or type(end) is not int or start != cursor or end <= start or end > token_count:
                parsed = []
                break
            parsed.append((start, end, piece.get("version")))
            cursor = end
        if parsed and cursor == token_count:
            tiled = parsed
    if tiled is None:
        note = "broken_intervals" if isinstance(pieces, list) and pieces else "attempt"
        raw = None if note == "broken_intervals" else meta.get("weight_version")
        if note == "attempt" and raw is None:
            note = "absent"
        tiled = [(0, token_count, raw)]
        evidence = False
        unknown_override = None if note == "attempt" else note
    else:
        evidence = True
        unknown_override = None
    segments = []
    for index, (start, end, raw) in enumerate(tiled):
        segments.append(
            {
                "start": offset + start,
                "end": offset + end,
                "raw_version": None if unknown_override else raw,
                "request_id": request_id,
                "attempt_id": attempt_id,
                "piece": index,
                "evidence": "intervals" if evidence else unknown_override or "attempt",
            }
        )
    return segments


def stamp_commit(patch: dict[str, Any]) -> None:
    """Remember the publication head at the moment a response is committed."""

    record = patch.setdefault(METADATA_KEY, {"segments": []})
    head = read_head()
    record["commit_published"] = None if head is None else head["published"]
    record["commit_run"] = None if head is None else head["run_id"]


def begin_sample(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "turns": [], "segments": []}


def take_turn(sample: dict[str, Any], patch: dict[str, Any] | None, *, absolute_start: int, turn: int) -> None:
    """Move one committed response onto absolute sample-token coordinates."""

    record = (patch or {}).get(METADATA_KEY) or {}
    sample["turns"].append(
        {
            "turn": turn,
            "commit_published": record.get("commit_published"),
            "commit_run": record.get("commit_run"),
        }
    )
    for segment in record.get("segments") or []:
        sample["segments"].append(
            {
                "token_start": absolute_start + int(segment["start"]),
                "token_end": absolute_start + int(segment["end"]),
                "turn": turn,
                "request_id": segment.get("request_id"),
                "attempt_id": segment.get("attempt_id"),
                "piece": segment.get("piece"),
                "evidence": segment.get("evidence"),
                "raw_version": segment.get("raw_version"),
            }
        )


def pack_record(record: dict[str, Any]) -> list[int]:
    return list(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def unpack_record(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    return json.loads(bytes(int(item) for item in payload).decode("utf-8"))


def _mask_values(mask: Any) -> list[int]:
    values = mask.tolist() if hasattr(mask, "tolist") else list(mask)
    if any(value not in (0, 1) for value in values):
        raise ValueError("provenance accounting needs a 0/1 loss mask")
    return [int(value) for value in values]


def _span(seqs: list[int]) -> int | None:
    if not seqs:
        return None
    return max(seqs) - min(seqs)


def attribute_row(
    record: dict[str, Any],
    total_length: int,
    response_length: int,
    mask: Any,
    head: dict[str, Any] | None,
) -> dict[str, Any]:
    """Count trained tokens only. Prompt and tool-observation positions stay out.

    ``trajectory_version_span`` is the published-sequence gap across every known
    trained token in the sample, including later turns produced after a tool
    wait. ``attempt_version_span`` is the gap inside a single backend attempt.
    ``lag_after_final_commit`` is how many publishes happened after the last
    response was committed and before this training load. Each turn also reports
    ``publishes_since_commit``, so a tool wait is visible on the turn that had
    already finished.
    """

    values = _mask_values(mask)
    if len(values) != response_length or not 0 <= response_length <= total_length:
        raise ValueError("loss mask must match the response suffix")
    prompt = total_length - response_length
    trained = sum(values)
    covered = [0] * response_length
    pieces: list[dict[str, Any]] = []
    known = 0
    age_tokens = 0
    age_sum = 0
    future = 0
    unknown_reasons: dict[str, int] = {}
    by_attempt: dict[str, list[int]] = {}
    by_turn: dict[int, list[int]] = {}
    error = None
    segments = list(record.get("segments") or [])
    segments.sort(key=lambda item: (item["token_start"], item["token_end"]))
    previous = prompt
    for segment in segments:
        start, end = segment["token_start"], segment["token_end"]
        if type(start) is not int or type(end) is not int or start < previous or end <= start or end > total_length:
            error = "segment_overlap_or_out_of_range"
            break
        previous = end
        local_start = max(0, start - prompt)
        local_end = min(response_length, end - prompt)
        count = sum(values[local_start:local_end]) if local_end > local_start else 0
        for index in range(local_start, local_end):
            if values[index]:
                covered[index] += 1
        source = resolve_source(segment.get("raw_version"), head)
        if segment.get("evidence") in ("absent", "broken_intervals"):
            source = {**source, "seq": None, "unknown": segment["evidence"]}
        seq = source["seq"]
        age = None
        if count and seq is not None and head is not None:
            delta = head["published"] - seq
            if delta >= 0:
                known += count
                age = delta
                age_tokens += count
                age_sum += count * delta
                by_attempt.setdefault(str(segment.get("attempt_id")), []).append(seq)
                by_turn.setdefault(int(segment["turn"]), []).append(seq)
            else:
                future += count
                unknown_reasons["future_publication"] = unknown_reasons.get("future_publication", 0) + count
        elif count:
            known_reason = source["unknown"] or "unknown"
            unknown_reasons[known_reason] = unknown_reasons.get(known_reason, 0) + count
        pieces.append({**segment, "trained_tokens": count, "source_seq": seq, "unknown": source["unknown"], "age": age})
    if error is None and any(item > 1 for item in covered):
        error = "segment_overlap_or_out_of_range"
    uncovered = 0 if error else sum(1 for flag, hits in zip(values, covered, strict=True) if flag and hits == 0)
    if uncovered:
        unknown_reasons["no_segment"] = unknown_reasons.get("no_segment", 0) + uncovered
    if error:
        known = age_tokens = age_sum = future = 0
        unknown_reasons = {"bad_record": trained}
        by_attempt = {}
        by_turn = {}
        pieces = []
    turns = []
    for turn in record.get("turns") or []:
        commit = turn.get("commit_published")
        same_run = (
            head is not None
            and commit is not None
            and turn.get("commit_run") == head["run_id"]
            and head["published"] >= commit
        )
        seqs = by_turn.get(turn["turn"], [])
        turns.append(
            {
                "turn": turn["turn"],
                "commit_published": commit,
                "source_min": min(seqs) if seqs else None,
                "source_max": max(seqs) if seqs else None,
                "publishes_since_commit": head["published"] - commit if same_run else None,
            }
        )
    for index, turn in enumerate(turns):
        nxt = turns[index + 1] if index + 1 < len(turns) else None
        gap = None
        if (
            nxt is not None
            and turn["commit_published"] is not None
            and nxt["commit_published"] is not None
            and record.get("turns", [])[index].get("commit_run") == record.get("turns", [])[index + 1].get("commit_run")
        ):
            gap = nxt["commit_published"] - turn["commit_published"]
        turn["publishes_before_next_commit"] = gap
    final = turns[-1] if turns else None
    lag = None if final is None else final["publishes_since_commit"]
    attempt_span = max((_span(seqs) or 0 for seqs in by_attempt.values()), default=None)
    if attempt_span is None and not by_attempt:
        attempt_span = None
    all_seqs = [seq for seqs in by_attempt.values() for seq in seqs]
    return {
        "session_id": record.get("session_id"),
        "trained_token_count": trained,
        "known_token_count": known,
        "unknown_fraction": None if trained == 0 else (trained - known - future) / trained,
        "weighted_version_age": None if age_tokens == 0 else age_sum / age_tokens,
        "age_token_sum": age_sum,
        "age_token_count": age_tokens,
        "future_token_count": future,
        "oldest_age": None if age_tokens == 0 else max(piece["age"] for piece in pieces if piece["age"] is not None),
        "newest_age": None if age_tokens == 0 else min(piece["age"] for piece in pieces if piece["age"] is not None),
        "attempt_version_span": attempt_span if by_attempt else None,
        "trajectory_version_span": _span(all_seqs),
        "lag_after_final_commit": lag,
        "unknown_reasons": unknown_reasons,
        "record_error": error,
        "turns": turns,
        "segments": pieces,
        "consumption": head,
    }


def combine(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token counts. Ages are recomputed from those sums, not averaged."""

    trained = sum(report["trained_token_count"] for report in reports)
    known = sum(report["known_token_count"] for report in reports)
    future = sum(report["future_token_count"] for report in reports)
    age_tokens = sum(report["age_token_count"] for report in reports)
    age_sum = sum(report["age_token_sum"] for report in reports)

    def _max(name: str) -> int | None:
        values = [report[name] for report in reports if report.get(name) is not None]
        return max(values) if values else None

    reasons: dict[str, int] = {}
    for report in reports:
        for key, count in report.get("unknown_reasons", {}).items():
            reasons[key] = reasons.get(key, 0) + count
    return {
        "rows": len(reports),
        "trained_token_count": trained,
        "known_token_count": known,
        "unknown_fraction": None if trained == 0 else (trained - known - future) / trained,
        "weighted_version_age": None if age_tokens == 0 else age_sum / age_tokens,
        "attempt_version_span_max": _max("attempt_version_span"),
        "trajectory_version_span_max": _max("trajectory_version_span"),
        "lag_after_final_commit_max": _max("lag_after_final_commit"),
        "unknown_reasons": reasons,
        "record_errors": sum(1 for report in reports if report.get("record_error")),
    }


def consume_loaded_batch(rollout_data: dict[str, Any], *, rank: int = 0) -> dict[str, Any] | None:
    """Attribute a CPU training batch and append a diagnostic file.

    The numeric provenance column is removed so the actor loss never sees it.
    """

    payloads = rollout_data.get(RECORD_FIELD)
    if payloads is None:
        return None
    if hasattr(payloads, "unbind"):
        payloads = list(payloads.unbind())
    head = read_head()
    reports = []
    for index, payload in enumerate(payloads):
        total = int(rollout_data["total_lengths"][index])
        response = int(rollout_data["response_lengths"][index])
        mask = rollout_data["loss_masks"][index]
        try:
            record = unpack_record(payload)
            if not isinstance(record, dict):
                raise ValueError("provenance record must be an object")
            reports.append(attribute_row(record, total, response, mask, head))
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError):
            try:
                values = _mask_values(mask)
                trained = sum(values) if len(values) == response else 0
            except (ValueError, TypeError):
                trained = 0
            reports.append(
                {
                    "session_id": None,
                    "trained_token_count": trained,
                    "known_token_count": 0,
                    "unknown_fraction": None if trained == 0 else 1,
                    "weighted_version_age": None,
                    "age_token_sum": 0,
                    "age_token_count": 0,
                    "future_token_count": 0,
                    "oldest_age": None,
                    "newest_age": None,
                    "attempt_version_span": None,
                    "trajectory_version_span": None,
                    "lag_after_final_commit": None,
                    "unknown_reasons": {"bad_record": trained} if trained else {},
                    "record_error": "bad_record",
                    "turns": [],
                    "segments": [],
                    "consumption": head,
                }
            )
    summary = combine(reports)
    path = _directory() / "diagnostics" / f"rank-{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps({"rank": rank, **report}, ensure_ascii=False) + "\n")
    _atomic_write(path.with_suffix(".summary.json"), {"rank": rank, **summary})
    rollout_data.pop(RECORD_FIELD, None)
    return summary
