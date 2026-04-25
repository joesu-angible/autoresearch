"""End-to-end tournament flow: propose → run (dry) → promote.

Covers the data-model wiring and CLI orchestration without launching real
subprocesses. Real-run integration is exercised manually via the smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_loop import tournament
from research_loop.candidate import (
    Candidate,
    Decision,
    Outcome,
    append_history,
    new_round_id,
    read_decisions,
    read_history,
    read_outcomes,
)


@pytest.fixture
def tmp_history(tmp_path: Path, monkeypatch):
    """Redirect the module-level history path to a temp file."""
    p = tmp_path / "history.jsonl"
    monkeypatch.setattr(tournament, "HISTORY_PATH", p)
    return p


def test_round_id_format():
    rid = new_round_id()
    assert rid.startswith("r")
    assert len(rid) >= 12  # rXXXXXXXXXX-yyyyyy


def test_propose_writes_three_candidates_with_same_round_id(tmp_history):
    rc = tournament.cmd_propose("student_v2", "test hypothesis", baseline_only=False)
    assert rc == 0
    candidates = list(read_history(tmp_history))
    assert len(candidates) == 3
    kinds = sorted(c.kind for c in candidates)
    assert kinds == ["A", "AB", "B"]
    rids = {c.round_id for c in candidates}
    assert len(rids) == 1  # all three share one round_id


def test_propose_baseline_only_writes_just_a(tmp_history):
    rc = tournament.cmd_propose("student_v2", "baseline run", baseline_only=True)
    assert rc == 0
    candidates = list(read_history(tmp_history))
    assert len(candidates) == 1
    assert candidates[0].kind == "A"


def test_propose_unknown_target_returns_error(tmp_history):
    assert tournament.cmd_propose("not_a_target", "x", baseline_only=False) == 2


# ---------------------------------------------------------------------------
# --variants mode (issue #9 Goal 3)
# ---------------------------------------------------------------------------

def _write_variants_file(path: Path, n: int) -> Path:
    """Write a JSONL file with `n` minimal-but-valid variant entries."""
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "hypothesis": f"variant {i}",
            "expected_metric": f"combined +0.00{i}",
            "changed_files": ["student_finetune/train_v2.py"],
            "risks": ["regress recall@1"],
            "rollback": "combined < incumbent - 0.005",
            "patch": f"--- a/x\n+++ b/x\n@@\n+v{i}\n",
        }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_propose_variants_writes_a_plus_n_with_one_round_id(tmp_history, tmp_path):
    vfile = _write_variants_file(tmp_path / "v.jsonl", n=4)
    rc = tournament.cmd_propose("student_v2", "sweep", baseline_only=False, variants=vfile)
    assert rc == 0
    candidates = list(read_history(tmp_history))
    assert len(candidates) == 5  # 1 A + 4 variants
    kinds = [c.kind for c in candidates]
    assert kinds.count("A") == 1
    assert kinds.count("B") == 4
    assert "AB" not in kinds  # variants mode must not auto-synthesize AB
    rids = {c.round_id for c in candidates}
    assert len(rids) == 1  # all share one round


def test_propose_variants_fills_parent_incumbent_id(tmp_history, tmp_path):
    vfile = _write_variants_file(tmp_path / "v.jsonl", n=2)
    rc = tournament.cmd_propose("student_v2", "sweep", baseline_only=False, variants=vfile)
    assert rc == 0
    candidates = list(read_history(tmp_history))
    a = next(c for c in candidates if c.kind == "A")
    bs = [c for c in candidates if c.kind == "B"]
    assert all(b.parent_incumbent_id == a.id for b in bs)


def test_propose_variants_distinct_candidate_ids(tmp_history, tmp_path):
    vfile = _write_variants_file(tmp_path / "v.jsonl", n=3)
    tournament.cmd_propose("student_v2", "sweep", baseline_only=False, variants=vfile)
    bs = [c for c in read_history(tmp_history) if c.kind == "B"]
    assert len({b.id for b in bs}) == 3  # uniqueness by id


def test_propose_variants_and_baseline_only_mutually_exclusive(tmp_history, tmp_path):
    vfile = _write_variants_file(tmp_path / "v.jsonl", n=1)
    rc = tournament.cmd_propose("student_v2", "x", baseline_only=True, variants=vfile)
    assert rc == 2  # programmatic guard


def test_cli_variants_and_baseline_only_argparse_error(tmp_history, tmp_path, capsys):
    vfile = _write_variants_file(tmp_path / "v.jsonl", n=1)
    with pytest.raises(SystemExit):
        tournament.main([
            "propose", "--target", "student_v2",
            "--baseline-only", "--variants", str(vfile),
        ])
    err = capsys.readouterr().err
    assert "not allowed with" in err or "mutually exclusive" in err


def test_propose_variants_empty_file_writes_only_a_with_warning(tmp_history, tmp_path, capsys):
    vfile = tmp_path / "v.jsonl"
    vfile.write_text("\n\n")  # only blanks
    rc = tournament.cmd_propose("student_v2", "x", baseline_only=False, variants=vfile)
    assert rc == 0
    candidates = list(read_history(tmp_history))
    assert len(candidates) == 1 and candidates[0].kind == "A"
    assert "no entries" in capsys.readouterr().err


def test_outcome_roundtrip(tmp_history):
    o = Outcome(
        candidate_id="abc",
        round_id="r123",
        target="student_v2",
        status="success",
        metrics={"combined": 0.86, "recall_1": 0.91, "productness_neg_acc": 0.86},
        elapsed_seconds=120.0,
        log_path="/tmp/x.log",
        metrics_json_path="/tmp/m.json",
    )
    append_history(tmp_history, o)
    loaded = list(read_outcomes(tmp_history))
    assert len(loaded) == 1
    assert loaded[0].candidate_id == "abc"
    assert loaded[0].metrics["combined"] == 0.86


def test_outcomes_filter_by_round_id(tmp_history):
    for rid in ("r1", "r2", "r1"):
        append_history(tmp_history, Outcome(
            candidate_id=f"c{rid}", round_id=rid, target="student_v2",
            status="success", metrics={"combined": 0.8},
            elapsed_seconds=1.0, log_path="x", metrics_json_path="y",
        ))
    r1 = list(read_outcomes(tmp_history, round_id="r1"))
    assert len(r1) == 2


def test_dry_run_pipeline_writes_outcome_no_decision(tmp_history, monkeypatch):
    """propose → run (dry) → no promote → outcomes recorded as noop."""
    # Make sure adapters can be imported without side effects
    rc = tournament.cmd_run_round(
        "student_v2", "dry-run smoke",
        epochs=None, baseline_only=True, dry_run=True,
    )
    # dry-run doesn't promote, so cmd_run_round returns 0 from the dry-run branch
    assert rc == 0
    outcomes = list(read_outcomes(tmp_history))
    assert len(outcomes) == 1
    assert outcomes[0].status == "noop"
    # No decision because we short-circuited promote
    decisions = list(read_decisions(tmp_history))
    assert decisions == []


def test_promote_with_synthetic_outcomes_writes_decision(tmp_history):
    """Hand-craft a round with a clear winner; verify decide() + Decision write."""
    rid = "r-test"
    a = Candidate(
        kind="A", target="student_v2", round_id=rid,
        hypothesis="incumbent", expected_metric="0.0",
        changed_files=[], risks=[], rollback="N/A", patch="",
    )
    b = Candidate(
        kind="B", target="student_v2", round_id=rid,
        hypothesis="strong proposal with enough text",
        expected_metric="combined +0.01",
        changed_files=["student_finetune/train_v2.py"],
        risks=["may slow"], rollback="combined < 0.85",
        patch="--- a\n+++ b\n",
    )
    append_history(tmp_history, a)
    append_history(tmp_history, b)
    append_history(tmp_history, Outcome(
        candidate_id=a.id, round_id=rid, target="student_v2",
        status="success",
        metrics={"combined": 0.860, "recall_1": 0.900, "mean_cosine": 0.81,
                 "productness_pos_acc": 0.99, "productness_neg_acc": 0.85},
        elapsed_seconds=100.0, log_path="x", metrics_json_path="y",
    ))
    append_history(tmp_history, Outcome(
        candidate_id=b.id, round_id=rid, target="student_v2",
        status="success",
        metrics={"combined": 0.875, "recall_1": 0.905, "mean_cosine": 0.82,
                 "productness_pos_acc": 0.99, "productness_neg_acc": 0.86},
        elapsed_seconds=110.0, log_path="x", metrics_json_path="y",
    ))

    rc = tournament.cmd_promote(rid)
    assert rc == 0
    decisions = list(read_decisions(tmp_history))
    assert len(decisions) == 1
    d = decisions[0]
    assert d.round_id == rid
    assert d.winner_id == b.id  # B beat A on combined + neg_acc both up
    assert d.promote is True


def test_promote_blocked_by_neg_acc_regression(tmp_history):
    """Even with combined improvement, productness_neg_acc regression must veto."""
    rid = "r-veto"
    a = Candidate(kind="A", target="student_v2", round_id=rid,
                  hypothesis="incumbent", expected_metric="0",
                  changed_files=[], risks=[], rollback="N/A", patch="")
    b = Candidate(kind="B", target="student_v2", round_id=rid,
                  hypothesis="proposal that tanks negatives",
                  expected_metric="combined +0.01",
                  changed_files=["student_finetune/train_v2.py"],
                  risks=["neg regress"], rollback="combined < 0.85",
                  patch="--- a\n+++ b\n")
    append_history(tmp_history, a)
    append_history(tmp_history, b)
    append_history(tmp_history, Outcome(
        candidate_id=a.id, round_id=rid, target="student_v2",
        status="success",
        metrics={"combined": 0.860, "recall_1": 0.900, "mean_cosine": 0.81,
                 "productness_pos_acc": 0.99, "productness_neg_acc": 0.85},
        elapsed_seconds=1.0, log_path="x", metrics_json_path="y",
    ))
    append_history(tmp_history, Outcome(
        candidate_id=b.id, round_id=rid, target="student_v2",
        status="success",
        metrics={"combined": 0.880, "recall_1": 0.905, "mean_cosine": 0.84,
                 "productness_pos_acc": 0.99, "productness_neg_acc": 0.80},  # -5 pts
        elapsed_seconds=1.0, log_path="x", metrics_json_path="y",
    ))

    rc = tournament.cmd_promote(rid)
    assert rc == 0
    d = next(read_decisions(tmp_history))
    assert d.winner_kind == "A"
    assert d.promote is False


def test_log_row_writes_results_v2_tsv(tmp_path: Path, monkeypatch):
    """The adapter's log_row must write to results_v2.tsv with the expected columns."""
    from research_loop.targets._base import TargetAdapter, V2_RESULTS_COLUMNS

    class FakeTarget(TargetAdapter):
        name = "fake_v2"

    target = FakeTarget()
    target.RESULTS_TSV = tmp_path / "results_v2.tsv"

    candidate = Candidate(
        kind="A", target="student_v2", round_id="r1",
        hypothesis="baseline run", expected_metric="0",
        changed_files=[], risks=[], rollback="N/A", patch="",
    )
    from research_loop.targets._base import TrainOutcome
    outcome = TrainOutcome(
        candidate_id=candidate.id, metrics={"combined": 0.86, "recall_1": 0.9, "productness_neg_acc": 0.85},
        elapsed_seconds=120.0, status="success", log_path=Path("x"),
        metrics_json_path=Path("y"),
    )
    target.log_row(candidate, outcome)
    assert target.RESULTS_TSV.exists()
    lines = target.RESULTS_TSV.read_text().splitlines()
    assert lines[0].split("\t") == list(V2_RESULTS_COLUMNS)  # header
    assert "0.860000" in lines[1]
    assert candidate.id in lines[1]


def test_log_row_refuses_v1_path(tmp_path: Path, monkeypatch):
    from research_loop.targets._base import TargetAdapter, TrainOutcome

    class BadTarget(TargetAdapter):
        name = "bad"

    bad = BadTarget()
    bad.RESULTS_TSV = tmp_path / "results.tsv"  # V1 name → must be rejected
    candidate = Candidate(
        kind="A", target="student_v2", round_id="r",
        hypothesis="", expected_metric="0",
        changed_files=[], risks=[], rollback="N/A", patch="",
    )
    outcome = TrainOutcome(
        candidate_id=candidate.id, metrics={}, elapsed_seconds=1.0,
        status="success", log_path=Path("x"), metrics_json_path=Path("y"),
    )
    with pytest.raises(ValueError, match="V1 log path"):
        bad.log_row(candidate, outcome)
