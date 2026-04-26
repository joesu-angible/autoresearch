"""Tests for OutcomeStartedRecord + find_unfinished_candidates (issue #14 T1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_loop.candidate import (
    Candidate,
    Outcome,
    OutcomeStartedRecord,
    append_history,
    find_unfinished_candidates,
    read_outcomes_started,
)


def _candidate(cid: str, kind: str = "B", round_id: str = "r1") -> Candidate:
    return Candidate(
        kind=kind,
        target="student_v2",
        round_id=round_id,
        hypothesis="x",
        expected_metric="combined +0.005",
        changed_files=["student_finetune/train_v2.py"] if kind != "A" else [],
        risks=["x"] if kind != "A" else [],
        rollback="combined < incumbent - 0.005" if kind != "A" else "N/A",
        patch="--- a/x\n+++ b/x\n@@\n+v\n" if kind != "A" else "",
        id=cid,
    )


def _started(cid: str, run_id: str = "run-A", round_id: str = "r1",
             kind: str = "B", pass_index: int = 1) -> OutcomeStartedRecord:
    return OutcomeStartedRecord(
        candidate_id=cid, round_id=round_id, target="student_v2",
        pass_index=pass_index, kind=kind, run_id=run_id,
    )


def _outcome(cid: str, round_id: str = "r1", status: str = "success") -> Outcome:
    return Outcome(
        candidate_id=cid, round_id=round_id, target="student_v2",
        status=status, metrics={"combined": 0.86},
        elapsed_seconds=10.0, log_path="x", metrics_json_path="y",
    )


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

def test_outcome_started_record_roundtrip():
    rec = _started("c1")
    line = rec.to_jsonl()
    parsed = OutcomeStartedRecord.from_jsonl(line)
    assert parsed.candidate_id == "c1"
    assert parsed.run_id == "run-A"
    assert parsed.kind == "B"
    assert parsed.record_type == "outcome_started"


def test_record_type_field_correct():
    rec = _started("c1")
    assert rec.record_type == "outcome_started"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_read_outcomes_started_filters_by_run_id(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    append_history(h, _started("c1", run_id="run-A"))
    append_history(h, _started("c2", run_id="run-B"))
    append_history(h, _started("c3", run_id="run-A"))
    a_records = list(read_outcomes_started(h, run_id="run-A"))
    assert len(a_records) == 2
    assert {r.candidate_id for r in a_records} == {"c1", "c3"}


def test_read_outcomes_started_skips_other_record_types(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    append_history(h, _candidate("c1"))
    append_history(h, _started("c1"))
    append_history(h, _outcome("c1"))
    started = list(read_outcomes_started(h))
    assert len(started) == 1
    assert started[0].candidate_id == "c1"


# ---------------------------------------------------------------------------
# find_unfinished_candidates — the core state machine
# ---------------------------------------------------------------------------

def test_find_unfinished_returns_candidates_with_started_no_outcome(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    # c1: started + outcome → finished
    # c2: started + no outcome → UNFINISHED
    # c3: never started → not tracked
    append_history(h, _candidate("c1"))
    append_history(h, _candidate("c2"))
    append_history(h, _candidate("c3"))
    append_history(h, _started("c1", run_id="run-A"))
    append_history(h, _outcome("c1"))
    append_history(h, _started("c2", run_id="run-A"))
    # c3 never started

    unfinished = find_unfinished_candidates(h, run_id="run-A")
    assert [c.id for c in unfinished] == ["c2"]


def test_find_unfinished_filters_by_run_id(tmp_path: Path):
    """Different runs should not bleed into each other."""
    h = tmp_path / "history.jsonl"
    append_history(h, _candidate("c1"))
    append_history(h, _candidate("c2"))
    append_history(h, _started("c1", run_id="run-A"))
    append_history(h, _started("c2", run_id="run-B"))
    # c1 has no outcome, c2 has no outcome — both unfinished but different runs

    a_unfinished = find_unfinished_candidates(h, run_id="run-A")
    b_unfinished = find_unfinished_candidates(h, run_id="run-B")
    assert [c.id for c in a_unfinished] == ["c1"]
    assert [c.id for c in b_unfinished] == ["c2"]


def test_find_unfinished_preserves_start_order(tmp_path: Path):
    """Re-run order must match original start order for deterministic recovery."""
    h = tmp_path / "history.jsonl"
    for cid in ("cA", "cB", "cC"):
        append_history(h, _candidate(cid))
    # Started in order A, B, C — none completed
    append_history(h, _started("cA", run_id="run-A"))
    append_history(h, _started("cB", run_id="run-A"))
    append_history(h, _started("cC", run_id="run-A"))

    unfinished = find_unfinished_candidates(h, run_id="run-A")
    assert [c.id for c in unfinished] == ["cA", "cB", "cC"]


def test_find_unfinished_tolerates_duplicate_started_records(tmp_path: Path):
    """Resume → recrash → resume produces multiple started records for one candidate.
    Pairing should still match against the single outcome (or absence)."""
    h = tmp_path / "history.jsonl"
    append_history(h, _candidate("c1"))
    append_history(h, _started("c1", run_id="run-A"))   # first attempt
    append_history(h, _started("c1", run_id="run-A"))   # resume re-marks
    # No outcome → still unfinished (de-duplicated to single entry)

    unfinished = find_unfinished_candidates(h, run_id="run-A")
    assert [c.id for c in unfinished] == ["c1"]


def test_find_unfinished_backward_compat_no_started_records(tmp_path: Path):
    """A history.jsonl from before this PR (no outcome_started records) must
    return [] — nothing was tracked, so nothing is reportable as unfinished.
    Resume falls back to "start a fresh pass" in this case."""
    h = tmp_path / "history.jsonl"
    append_history(h, _candidate("c1"))
    append_history(h, _outcome("c1"))
    append_history(h, _candidate("c2"))
    # c2 has neither started nor outcome — but no started anywhere

    unfinished = find_unfinished_candidates(h, run_id="run-A")
    assert unfinished == []


def test_find_unfinished_empty_history(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    assert find_unfinished_candidates(h, run_id="run-A") == []


def test_find_unfinished_all_completed(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    for cid in ("c1", "c2", "c3"):
        append_history(h, _candidate(cid))
        append_history(h, _started(cid, run_id="run-A"))
        append_history(h, _outcome(cid))

    assert find_unfinished_candidates(h, run_id="run-A") == []


# ---------------------------------------------------------------------------
# Backward-compat: existing readers ignore the new record_type
# ---------------------------------------------------------------------------

def test_existing_readers_ignore_outcome_started(tmp_path: Path):
    """read_outcomes / read_history / etc must skip outcome_started records."""
    from research_loop.candidate import read_history, read_outcomes
    h = tmp_path / "history.jsonl"
    append_history(h, _candidate("c1"))
    append_history(h, _started("c1"))
    append_history(h, _outcome("c1"))

    assert [c.id for c in read_history(h)] == ["c1"]
    assert [o.candidate_id for o in read_outcomes(h)] == ["c1"]
