"""Crash-matrix tests for autoreason --resume (issue #14 T4).

Each test simulates a different crash point in the autoreason loop and
verifies that --resume picks up correctly with no data loss + clean tree.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_loop import tournament
from research_loop.candidate import (
    Candidate,
    Decision,
    Outcome,
    OutcomeStartedRecord,
    append_history,
    find_unfinished_candidates,
    read_decisions,
    read_history,
    read_outcomes,
    read_outcomes_started,
)


# Mirror prompts so the same _mock_agent_client_factory works
CRITIC_RAW = """SUMMARY: stub.

PROBLEMS:
- problem 1
"""
AUTHOR_B_RAW = """RATIONALE: stub.

```diff
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -1 +1 @@
-# stub
+# stub-v2
```
"""
SYNTH_RAW = """RATIONALE: stub synthesis.

```diff
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -1 +1 @@
-# stub
+# stub-ab
```
"""


@pytest.fixture
def history_in_tmp(tmp_path: Path, monkeypatch):
    p = tmp_path / "history.jsonl"
    monkeypatch.setattr(tournament, "HISTORY_PATH", p)
    return p


@pytest.fixture
def runs_dir_in_tmp(tmp_path: Path, monkeypatch):
    rd = tmp_path / "runs"
    rd.mkdir()
    monkeypatch.setattr(tournament, "RUNS_DIR", rd)
    return rd


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "student_finetune").mkdir()
    (repo / "student_finetune" / "train_v2.py").write_text("# stub\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    monkeypatch.setattr(tournament, "REPO_ROOT", repo)
    return repo


def _mock_factory():
    def make_client(role: str):
        client = MagicMock()
        def call_fn(system, user, *, temperature=0.8, max_tokens=None, timeout=None):
            s = system.lower()
            if "identify concrete problems" in s:
                return CRITIC_RAW
            if "produce a unified diff that addresses" in s:
                return AUTHOR_B_RAW
            if "conservative synthesis" in s:
                return SYNTH_RAW
            raise AssertionError(f"Unrecognized system prompt: {system[:80]}")
        client.call.side_effect = call_fn
        client.name = f"mock-{role}"
        return client
    return make_client


def _patch_train(adapter_cls, monkeypatch, payloads):
    """Each call to .train() pops the next payload (dict) from `payloads`.
    Payload keys: 'status', 'metrics', or 'raises' (Exception type to throw)."""
    from research_loop.targets._base import TrainOutcome
    iterator = iter(payloads)

    def fake_train(self, candidate, max_epochs=None, dry_run=False, log_path=None, max_seconds=None):
        try:
            payload = next(iterator)
        except StopIteration:
            payload = {"status": "success", "metrics": {"combined": 0.86, "recall_1": 0.90,
                       "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}}
        if "raises" in payload:
            raise payload["raises"]
        return TrainOutcome(
            candidate_id=candidate.id,
            metrics=payload.get("metrics", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 1.0),
            status=payload.get("status", "success"),
            log_path=Path("/dev/null"),
            metrics_json_path=self.METRICS_JSON,
            return_code=payload.get("return_code", 0),
        )
    monkeypatch.setattr(adapter_cls, "train", fake_train)


def _patch_log_row(adapter_cls, monkeypatch):
    monkeypatch.setattr(adapter_cls, "log_row", lambda self, c, o: None)


# ---------------------------------------------------------------------------
# Argparse / orchestration guards
# ---------------------------------------------------------------------------

def test_resume_mutex_with_target_returns_2(history_in_tmp, runs_dir_in_tmp, fake_repo, capsys):
    rc = tournament.main([
        "autoreason", "--target", "student_v2", "--resume", "run-X",
    ])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_resume_or_target_required(history_in_tmp, runs_dir_in_tmp, fake_repo, capsys):
    rc = tournament.main(["autoreason"])
    assert rc == 2
    assert "--target is required" in capsys.readouterr().err


def test_resume_missing_run_dir_returns_2(history_in_tmp, runs_dir_in_tmp, fake_repo, capsys):
    rc = tournament.cmd_autoreason(
        None, max_passes=1, convergence=2, max_seconds_per_candidate=None,
        hypothesis_seed="x", dry_run=False, resume="no-such-run",
        agent_client_factory=_mock_factory(),
    )
    assert rc == 2
    assert "no run dir" in capsys.readouterr().err


def test_resume_alive_pid_refuses(history_in_tmp, runs_dir_in_tmp, fake_repo, capsys):
    """If the prior runner's PID is still alive, refuse resume."""
    run_id = "run-alive"
    run_dir = runs_dir_in_tmp / run_id
    run_dir.mkdir()
    (run_dir / "autoreason.pid").write_text(str(os.getpid()))  # our own PID = alive
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id, "target": "student_v2",
        "config": {"target": "student_v2", "max_passes": 1, "convergence": 2},
    }))

    rc = tournament.cmd_autoreason(
        None, max_passes=1, convergence=2, max_seconds_per_candidate=None,
        hypothesis_seed="x", dry_run=False, resume=run_id,
        agent_client_factory=_mock_factory(),
    )
    assert rc == 2
    assert "still alive" in capsys.readouterr().err


def test_resume_no_config_block_refuses(history_in_tmp, runs_dir_in_tmp, fake_repo, capsys):
    """summary.json from before T2 has no `config` → cannot resume."""
    run_id = "run-old"
    run_dir = runs_dir_in_tmp / run_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id, "target": "student_v2",
    }))

    rc = tournament.cmd_autoreason(
        None, max_passes=1, convergence=2, max_seconds_per_candidate=None,
        hypothesis_seed="x", dry_run=False, resume=run_id,
        agent_client_factory=_mock_factory(),
    )
    assert rc == 2
    assert "no `config`" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Crash-matrix integration tests
# ---------------------------------------------------------------------------

def test_resume_after_mid_training_crash_completes_run(
    history_in_tmp, runs_dir_in_tmp, fake_repo, monkeypatch
):
    """The expensive crash: outcome_started written, train() raises mid-way,
    resume detects unfinished, re-runs the crashed candidate, completes the round."""
    from research_loop.targets.student_v2 import StudentV2Target

    # Pass 1: A succeeds, B succeeds, AB raises mid-train. Then resume:
    # AB re-run succeeds. Round 1 promote. Then converge logic kicks in.
    payloads_first_run = [
        {"status": "success", "metrics": {"combined": 0.86, "recall_1": 0.90,
            "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}},  # A
        {"status": "success", "metrics": {"combined": 0.85, "recall_1": 0.89,
            "mean_cosine": 0.80, "productness_neg_acc": 0.84, "productness_pos_acc": 0.99}},  # B
        {"raises": RuntimeError("simulated mid-training OOM")},  # AB crashes
    ]
    _patch_train(StudentV2Target, monkeypatch, payloads_first_run)
    _patch_log_row(StudentV2Target, monkeypatch)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")

    with pytest.raises(RuntimeError, match="OOM"):
        tournament.cmd_autoreason(
            "student_v2",
            max_passes=2, convergence=2, max_seconds_per_candidate=None,
            hypothesis_seed="t4-crash", dry_run=False,
            agent_client_factory=_mock_factory(),
        )

    # Capture the run_id from the runs dir
    [run_dir] = list(runs_dir_in_tmp.glob("run*"))
    run_id = run_dir.name

    # Verify state: 2 outcomes (A, B), 1 unfinished (AB)
    started = list(read_outcomes_started(history_in_tmp))
    outcomes = list(read_outcomes(history_in_tmp))
    assert len(started) == 3  # A, B, AB all started
    assert len(outcomes) == 2  # only A and B completed
    unfinished = find_unfinished_candidates(history_in_tmp, run_id=run_id)
    assert len(unfinished) == 1
    assert unfinished[0].kind == "AB"

    # PID may still point at our process (test-time); clear it so resume succeeds
    (run_dir / "autoreason.pid").write_text("999999999")  # nonexistent

    # Resume: AB re-runs successfully, round promotes, pass 2 starts.
    # Pass 2: A wins again (1 consecutive A); pass 1's promoted winner is A.
    # Make resume's AB succeed and pass 2 cleanly converge.
    a_wins_metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
                      "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    b_loses_metrics = {"combined": 0.85, "recall_1": 0.89, "mean_cosine": 0.80,
                       "productness_neg_acc": 0.84, "productness_pos_acc": 0.99}
    payloads_resume = [
        {"status": "success", "metrics": b_loses_metrics},  # AB resume re-run
        {"status": "success", "metrics": a_wins_metrics},   # pass 2 A
        {"status": "success", "metrics": b_loses_metrics},  # pass 2 B
        {"status": "success", "metrics": b_loses_metrics},  # pass 2 AB
    ]
    _patch_train(StudentV2Target, monkeypatch, payloads_resume)

    rc = tournament.cmd_autoreason(
        None, max_passes=2, convergence=2, max_seconds_per_candidate=None,
        hypothesis_seed="ignored", dry_run=False,
        agent_client_factory=_mock_factory(), resume=run_id,
    )
    assert rc == 0  # clean exit (max_passes reached or converged)

    # AB now has an outcome
    final_outcomes = list(read_outcomes(history_in_tmp))
    assert len(final_outcomes) >= 3  # A, B, AB from pass 1 (+ pass 2 candidates)

    # Round 1 has a decision now
    decisions = list(read_decisions(history_in_tmp))
    assert len(decisions) >= 1  # at least pass 1 decided

    # Working tree clean after resume
    res = subprocess.run(
        ["git", "status", "--porcelain"], cwd=fake_repo,
        capture_output=True, text=True, check=True,
    )
    assert not res.stdout.strip()


def test_resume_when_outcomes_done_but_no_decision_runs_promote(
    history_in_tmp, runs_dir_in_tmp, fake_repo, monkeypatch
):
    """Crash between last outcome write and promote: resume detects the
    pending round and runs promote."""
    from research_loop.targets.student_v2 import StudentV2Target

    # Pre-seed: 1 round, 3 candidates with outcomes, no decision, no started
    # records (to verify promote-only path doesn't depend on outcome_started).
    rid = "r-pending"
    run_id = "run-promote-pending"
    run_dir = runs_dir_in_tmp / run_id
    run_dir.mkdir()

    # Persist a config block
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id, "target": "student_v2",
        "config": {
            "target": "student_v2", "max_passes": 1, "convergence": 2,
            "max_seconds_per_candidate": None, "hypothesis_seed": "test",
            "dry_run": False, "llm_cli": None, "llm_model": None,
            "llm_provider": "auto",
        },
        "started_at": "2026-04-25T00:00:00Z",
    }))
    (run_dir / "autoreason.pid").write_text("999999999")  # dead

    # Three candidates + outcomes
    a = Candidate(
        kind="A", target="student_v2", round_id=rid,
        hypothesis="test", expected_metric="0.0",
        changed_files=[], risks=[], rollback="N/A", patch="",
    )
    b = Candidate(
        kind="B", target="student_v2", round_id=rid,
        hypothesis="b", expected_metric="combined +0.005",
        changed_files=["student_finetune/train_v2.py"],
        risks=["x"], rollback="combined < incumbent",
        patch="--- a/x\n+++ b/x\n@@\n+v\n",
    )
    ab = Candidate(
        kind="AB", target="student_v2", round_id=rid,
        hypothesis="ab", expected_metric="combined +0.003",
        changed_files=["student_finetune/train_v2.py"],
        risks=["x"], rollback="combined < incumbent",
        patch="--- a/x\n+++ b/x\n@@\n+ab\n",
    )
    for c in (a, b, ab):
        append_history(history_in_tmp, c)
        append_history(history_in_tmp, Outcome(
            candidate_id=c.id, round_id=rid, target="student_v2",
            status="success", metrics={"combined": 0.86, "recall_1": 0.90,
                "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
                if c.kind == "A" else
                {"combined": 0.85, "recall_1": 0.89, "mean_cosine": 0.80,
                 "productness_neg_acc": 0.84, "productness_pos_acc": 0.99},
            elapsed_seconds=1.0, log_path="x", metrics_json_path="y",
        ))

    _patch_log_row(StudentV2Target, monkeypatch)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")
    # No train() should be called — there's nothing unfinished
    monkeypatch.setattr(StudentV2Target, "train",
        lambda self, c, **kw: (_ for _ in ()).throw(AssertionError("train should not run")))

    rc = tournament.cmd_autoreason(
        None, max_passes=1, convergence=2, max_seconds_per_candidate=None,
        hypothesis_seed="x", dry_run=False, resume=run_id,
        agent_client_factory=_mock_factory(),
    )
    # max_passes=1 reached → max_passes_exhausted (rc=1) is the expected path
    # (round was promoted as part of resume, then loop exits since pass 1 is done)
    assert rc in (0, 1)

    decisions = list(read_decisions(history_in_tmp))
    assert len(decisions) == 1
    assert decisions[0].round_id == rid
