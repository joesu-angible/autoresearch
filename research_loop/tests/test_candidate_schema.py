"""T2 verification: Candidate dataclass + JSONL roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_loop.candidate import (
    Candidate,
    REQUIRED_FIELDS,
    append_history,
    read_history,
)


def _b_kwargs(**overrides):
    base = dict(
        kind="B",
        target="student_v2",
        hypothesis="add stronger augmentation",
        expected_metric="combined +0.005",
        changed_files=["student_finetune/train_v2.py"],
        risks=["may slow convergence"],
        rollback="combined regresses below 0.85",
        patch="--- a\n+++ b\n@@\n+pass\n",
        evidence_refs=["results_v2.tsv:row=12"],
    )
    base.update(overrides)
    return base


def test_b_candidate_roundtrip():
    c = Candidate(**_b_kwargs())
    line = c.to_jsonl()
    parsed = Candidate.from_jsonl(line)
    assert parsed.id == c.id
    assert parsed.kind == "B"
    assert parsed.changed_files == ["student_finetune/train_v2.py"]


def test_a_incumbent_must_have_empty_patch():
    Candidate(
        kind="A",
        target="student_v2",
        hypothesis="do nothing — current incumbent",
        expected_metric="combined unchanged",
        changed_files=[],
        risks=[],
        rollback="N/A — incumbent baseline",
        patch="",
    )
    with pytest.raises(ValueError, match="empty patch"):
        Candidate(**_b_kwargs(kind="A", patch="--- a\n+++ b\n", changed_files=[]))


def test_b_requires_non_empty_patch():
    with pytest.raises(ValueError, match="non-empty patch"):
        Candidate(**_b_kwargs(patch="   "))


def test_rollback_required():
    with pytest.raises(ValueError, match="rollback"):
        Candidate(**_b_kwargs(rollback=""))


def test_changed_files_required_for_non_a():
    with pytest.raises(ValueError, match="changed_files"):
        Candidate(**_b_kwargs(changed_files=[]))


def test_required_fields_validated_on_load():
    c = Candidate(**_b_kwargs())
    raw = json.loads(c.to_jsonl())
    raw.pop("hypothesis")
    with pytest.raises(ValueError, match="Missing required fields"):
        Candidate.from_jsonl(json.dumps(raw))


def test_required_fields_constant_matches_dataclass():
    c = Candidate(**_b_kwargs())
    serialized = json.loads(c.to_jsonl())
    for field_name in REQUIRED_FIELDS:
        assert field_name in serialized


def test_history_append_and_read(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    a = Candidate(
        kind="A",
        target="student_v2",
        hypothesis="incumbent",
        expected_metric="unchanged",
        changed_files=[],
        risks=[],
        rollback="N/A",
    )
    b = Candidate(**_b_kwargs())
    append_history(history, a)
    append_history(history, b)
    loaded = list(read_history(history))
    assert [c.kind for c in loaded] == ["A", "B"]
    assert loaded[1].id == b.id
