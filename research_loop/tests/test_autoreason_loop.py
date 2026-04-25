"""T6 verification: full autoreason convergence loop with mocked LLM + adapter.

Tests the orchestrator at `research_loop.tournament.cmd_autoreason`. We:
  - patch out the AgentClient to deterministic canned responses
  - patch out the adapter's train() to return scripted outcomes (so we
    control who wins each pass without real GPU)
  - patch out the patch applicator's V1-safety check by writing a
    benign no-op patch in the canned LLM responses
  - assert the loop converges at k=2 consecutive A wins, hits max-passes
    correctly, writes the expected record_type sequence to history.jsonl,
    and respects the timeout veto.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_loop import tournament
from research_loop.candidate import (
    read_critiques,
    read_decisions,
    read_history,
    read_outcomes,
    read_patch_proposals,
    read_syntheses,
)


# A diff that adds a comment to a file; applies and reverts cleanly.
NOOP_DIFF = (
    "diff --git a/student_finetune/train_v2.py b/student_finetune/train_v2.py\n"
    "--- a/student_finetune/train_v2.py\n"
    "+++ b/student_finetune/train_v2.py\n"
    "@@ -1,2 +1,3 @@\n"
    " USE_PRODUCTNESS_CLS = True\n"
    "+# autoreason-test marker\n"
    " PRODUCTNESS_CLS_WEIGHT = 0.02\n"
)

CRITIC_RAW = """# Summary
Plateau on combined; productness neg_acc stuck at 0.76.

# Problems
- Productness CLS weight may be too low to drive gradient.
"""

AUTHOR_B_RAW = (
    "# Rationale\nRaise productness weight to address plateau.\n\n"
    "# Diff\n```\n" + NOOP_DIFF + "\n```\n"
)

SYNTH_RAW = (
    "# Rationale\nHalved the proposed change.\n\n"
    "# Diff\n```\n" + NOOP_DIFF + "\n```\n"
)


@pytest.fixture
def history_in_tmp(monkeypatch, tmp_path: Path):
    """Redirect HISTORY_PATH so tests don't pollute the real history.jsonl."""
    p = tmp_path / "history.jsonl"
    monkeypatch.setattr(tournament, "HISTORY_PATH", p)
    return p


@pytest.fixture
def fake_repo(monkeypatch, tmp_path: Path) -> Path:
    """Tiny git repo containing a stub student_finetune/train_v2.py the
    NOOP_DIFF can apply to. Used as REPO_ROOT for the duration of the test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "student_finetune").mkdir()
    (repo / "student_finetune" / "train_v2.py").write_text(
        "USE_PRODUCTNESS_CLS = True\n"
        "PRODUCTNESS_CLS_WEIGHT = 0.02\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    monkeypatch.setattr(tournament, "REPO_ROOT", repo)
    return repo


def _mock_agent_client_factory():
    """Returns a per-role factory: cmd_autoreason calls factory("critic") etc.

    Each role gets its own MagicMock so we can verify they're independent.
    All three return canned role-appropriate text; the agents call
    client.call(system, user) and we dispatch by system-prompt substring.
    """
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


def _patch_adapter_train(scripted_outcomes: list[dict]):
    """Patch StudentV2Target.train() to return scripted TrainOutcomes in order.

    Each entry should be a dict with keys for TrainOutcome (status, metrics,
    elapsed_seconds, ...).
    """
    from research_loop.targets._base import TrainOutcome
    iterator = iter(scripted_outcomes)

    def fake_train(self, candidate, max_epochs=None, dry_run=False, log_path=None, max_seconds=None):
        try:
            payload = next(iterator)
        except StopIteration:
            payload = {"status": "success", "metrics": {"combined": 0.85, "recall_1": 0.9, "mean_cosine": 0.81, "productness_neg_acc": 0.85}}
        return TrainOutcome(
            candidate_id=candidate.id,
            metrics=payload.get("metrics", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 1.0),
            status=payload.get("status", "success"),
            log_path=Path("/dev/null"),
            metrics_json_path=self.METRICS_JSON,
            return_code=payload.get("return_code", 0),
        )
    return fake_train


def _stub_log_row(self, candidate, outcome):
    """Skip TSV writes during tests — we only care about history.jsonl."""
    return None


def test_autoreason_converges_at_two_consecutive_a_wins(history_in_tmp, fake_repo, monkeypatch):
    # Pass 1: A wins (B and AB are no better)
    # Pass 2: A wins again → converges
    a_wins_metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    b_loses_metrics = {"combined": 0.85, "recall_1": 0.89, "mean_cosine": 0.80, "productness_neg_acc": 0.84, "productness_pos_acc": 0.99}
    scripted = [
        a_wins_metrics, b_loses_metrics, b_loses_metrics,    # pass 1: A, B, AB
        a_wins_metrics, b_loses_metrics, b_loses_metrics,    # pass 2: A, B, AB
    ]
    fake_outcomes = [{"status": "success", "metrics": m} for m in scripted]

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    # Make adapter.METRICS_JSON point inside fake_repo so tests don't hit /data
    monkeypatch.setattr(
        StudentV2Target, "METRICS_JSON",
        fake_repo / "metrics_final_v2.json",
    )

    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=5, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc == 0  # converged

    decisions = list(read_decisions(history_in_tmp))
    assert len(decisions) == 2
    assert all(d.winner_kind == "A" for d in decisions)


def test_autoreason_records_full_audit_trail(history_in_tmp, fake_repo, monkeypatch):
    """Every pass writes critique + patch_proposal + synthesis + 3×candidate +
    3×outcome + 1×decision = 10 records. After 1 pass we expect that count."""
    metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = [{"status": "success", "metrics": metrics}] * 3

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(
        StudentV2Target, "METRICS_JSON",
        fake_repo / "metrics_final_v2.json",
    )

    # max_passes=1, convergence=2 → won't converge but exits cleanly after 1
    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc == 1  # max_passes exhausted

    # All record types present
    assert len(list(read_critiques(history_in_tmp))) == 1
    assert len(list(read_patch_proposals(history_in_tmp))) == 1
    assert len(list(read_syntheses(history_in_tmp))) == 1
    candidates = list(read_history(history_in_tmp))
    kinds = sorted(c.kind for c in candidates)
    assert kinds == ["A", "AB", "B"]
    outcomes = list(read_outcomes(history_in_tmp))
    assert len(outcomes) == 3
    assert len(list(read_decisions(history_in_tmp))) == 1


def test_autoreason_max_passes_returns_nonzero(history_in_tmp, fake_repo, monkeypatch):
    """When max_passes is hit without convergence, exit code is non-zero."""
    # B and AB always score better than A → A never wins
    a_metrics = {"combined": 0.80, "recall_1": 0.85, "mean_cosine": 0.75, "productness_neg_acc": 0.80, "productness_pos_acc": 0.99}
    b_metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81, "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = []
    for _ in range(3):
        fake_outcomes.extend([
            {"status": "success", "metrics": a_metrics},  # A
            {"status": "success", "metrics": b_metrics},  # B wins
            {"status": "success", "metrics": b_metrics},  # AB
        ])

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(
        StudentV2Target, "METRICS_JSON",
        fake_repo / "metrics_final_v2.json",
    )

    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=3, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc != 0  # never converged


def test_autoreason_dry_run_records_noop_outcomes(history_in_tmp, fake_repo, monkeypatch):
    """Dry-run skips real subprocess training but still drives the LLM loop."""
    # Have train() return noop outcomes when dry_run=True
    from research_loop.targets._base import TrainOutcome
    from research_loop.targets.student_v2 import StudentV2Target

    def fake_train(self, candidate, max_epochs=None, dry_run=False, log_path=None, max_seconds=None):
        assert dry_run, "Expected dry_run=True"
        return TrainOutcome(
            candidate_id=candidate.id, metrics={}, elapsed_seconds=0.0,
            status="noop", log_path=Path("/dev/null"),
            metrics_json_path=self.METRICS_JSON, return_code=0,
        )

    monkeypatch.setattr(StudentV2Target, "train", fake_train)
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(
        StudentV2Target, "METRICS_JSON",
        fake_repo / "metrics_final_v2.json",
    )

    # Dry-run cmd_autoreason
    # cmd_promote will fail because all outcomes are status="noop"
    # so we expect a non-zero exit but with audit records written
    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=True,
        agent_client_factory=_mock_agent_client_factory(),
    )
    # rc may be != 0 because no successful outcome → no decision possible
    # but we still want audit records to exist
    assert len(list(read_critiques(history_in_tmp))) == 1


def test_autoreason_refuses_dirty_working_tree(history_in_tmp, fake_repo, monkeypatch):
    # Make the tree dirty
    (fake_repo / "stray.txt").write_text("dirty")

    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc == 2  # refused
    # Nothing should have been written
    assert not history_in_tmp.exists() or history_in_tmp.read_text() == ""


def test_autoreason_dry_run_skips_dirty_check(history_in_tmp, fake_repo, monkeypatch):
    """Dry-run should not require a clean tree (it doesn't apply patches anyway)."""
    (fake_repo / "stray.txt").write_text("dirty")

    from research_loop.targets._base import TrainOutcome
    from research_loop.targets.student_v2 import StudentV2Target

    def fake_train(self, candidate, max_epochs=None, dry_run=False, log_path=None, max_seconds=None):
        return TrainOutcome(
            candidate_id=candidate.id, metrics={}, elapsed_seconds=0.0,
            status="noop", log_path=Path("/dev/null"),
            metrics_json_path=self.METRICS_JSON, return_code=0,
        )

    monkeypatch.setattr(StudentV2Target, "train", fake_train)
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(
        StudentV2Target, "METRICS_JSON",
        fake_repo / "metrics_final_v2.json",
    )

    # Should run without complaining about dirty tree
    tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=True,
        agent_client_factory=_mock_agent_client_factory(),
    )
    # At minimum, the critic should have run
    assert len(list(read_critiques(history_in_tmp))) == 1


def test_autoreason_writes_summary_json_each_pass(history_in_tmp, fake_repo, monkeypatch, tmp_path):
    """summary.json is written atomically at startup + after each pass + on exit.
    External agents (Hermes, Slack bots) read this single file to answer
    'how is training going?' without parsing logs.
    """
    metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
               "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = [{"status": "success", "metrics": metrics}] * 6  # 2 passes × 3

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")

    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(tournament, "RUNS_DIR", runs_dir)

    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=2, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc == 0  # converged at k=2

    # Exactly one run dir was created; summary.json present
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    summary_path = run_dirs[0] / "summary.json"
    assert summary_path.exists()

    # Final summary state should reflect convergence
    import json
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "converged"
    assert summary["target"] == "student_v2"
    assert summary["current_pass"] == 2
    assert summary["consecutive_a_wins"] >= 2
    assert summary["last_decision"]["winner_kind"] == "A"
    assert summary["best_so_far"]["combined"] == 0.86

    # CURRENT pointer file exists
    pointer = runs_dir / "student_v2_CURRENT.txt"
    assert pointer.exists()
    assert pointer.read_text() == run_dirs[0].name


def test_autoreason_writes_narrative_log_to_run_dir(history_in_tmp, fake_repo, monkeypatch, tmp_path):
    """run_dir/autoreason.log captures the narrative print() output. External
    tooling (Hermes, Slack bots) can tail this single file from the path
    summary.json points at."""
    metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
               "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = [{"status": "success", "metrics": metrics}] * 3

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")
    monkeypatch.setattr(tournament, "RUNS_DIR", tmp_path / "runs")

    tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )

    run_dirs = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    log_path = run_dirs[0] / "autoreason.log"
    assert log_path.exists()
    content = log_path.read_text()
    # Narrative milestones should be captured
    assert "autoreason pass 1/1" in content
    assert "critic:" in content
    assert "running A" in content


def test_autoreason_summary_status_max_passes_exhausted(history_in_tmp, fake_repo, monkeypatch, tmp_path):
    """When max_passes hits without convergence, summary.json shows that status."""
    a_metrics = {"combined": 0.80, "recall_1": 0.85, "mean_cosine": 0.75,
                 "productness_neg_acc": 0.80, "productness_pos_acc": 0.99}
    b_metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
                 "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = []
    for _ in range(3):
        fake_outcomes.extend([
            {"status": "success", "metrics": a_metrics},
            {"status": "success", "metrics": b_metrics},
            {"status": "success", "metrics": b_metrics},
        ])

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")
    monkeypatch.setattr(tournament, "RUNS_DIR", tmp_path / "runs")

    rc = tournament.cmd_autoreason(
        "student_v2",
        max_passes=3, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    assert rc != 0

    runs = list((tmp_path / "runs").iterdir())
    summary_path = next(d for d in runs if d.is_dir()) / "summary.json"
    import json
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "max_passes_exhausted"


def test_status_subcommand_renders_summary(history_in_tmp, fake_repo, monkeypatch, tmp_path, capsys):
    """`tournament status` reads the latest run's summary.json and prints
    a human/bot-readable block with all the headline fields."""
    metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
               "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = [{"status": "success", "metrics": metrics}] * 6

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")
    monkeypatch.setattr(tournament, "RUNS_DIR", tmp_path / "runs")

    tournament.cmd_autoreason(
        "student_v2",
        max_passes=2, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    capsys.readouterr()  # flush autoreason's own output

    rc = tournament.cmd_status(target="student_v2")
    assert rc == 0
    out = capsys.readouterr().out

    # Headline fields all present
    assert "autoreason status — student_v2" in out
    assert "converged" in out
    assert "Pass:" in out
    assert "consecutive A wins:" in out
    assert "Best so far:" in out
    assert "combined=0.8600" in out
    assert "Logs:" in out


def test_status_subcommand_with_no_runs_returns_2(monkeypatch, tmp_path, capsys):
    """`tournament status` without any runs surfaces a clear error + non-zero exit."""
    monkeypatch.setattr(tournament, "RUNS_DIR", tmp_path / "runs")
    rc = tournament.cmd_status(target="student_v2")
    assert rc == 2
    err = capsys.readouterr().err
    assert "No" in err  # "No runs directory" or "No autoreason run found"


def test_status_subcommand_resolves_explicit_run_id(history_in_tmp, fake_repo, monkeypatch, tmp_path, capsys):
    """`tournament status --run RUN_ID` skips the CURRENT.txt pointer."""
    metrics = {"combined": 0.86, "recall_1": 0.90, "mean_cosine": 0.81,
               "productness_neg_acc": 0.85, "productness_pos_acc": 0.99}
    fake_outcomes = [{"status": "success", "metrics": metrics}] * 6

    from research_loop.targets.student_v2 import StudentV2Target
    monkeypatch.setattr(StudentV2Target, "train", _patch_adapter_train(fake_outcomes))
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")
    monkeypatch.setattr(tournament, "RUNS_DIR", tmp_path / "runs")

    tournament.cmd_autoreason(
        "student_v2", max_passes=2, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=False,
        agent_client_factory=_mock_agent_client_factory(),
    )
    capsys.readouterr()

    runs = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    rid = runs[0].name
    rc = tournament.cmd_status(run_id=rid)
    assert rc == 0
    out = capsys.readouterr().out
    assert rid in out


def test_autoreason_per_role_factory_called_with_each_role(history_in_tmp, fake_repo, monkeypatch):
    """Per-role overrides: each role's factory call gets its own role string,
    so callers can route Critic / Author / Synthesizer to different CLIs/models.
    """
    from research_loop.targets._base import TrainOutcome
    from research_loop.targets.student_v2 import StudentV2Target

    def fake_train(self, candidate, max_epochs=None, dry_run=False, log_path=None, max_seconds=None):
        return TrainOutcome(
            candidate_id=candidate.id, metrics={}, elapsed_seconds=0.0,
            status="noop", log_path=Path("/dev/null"),
            metrics_json_path=self.METRICS_JSON, return_code=0,
        )
    monkeypatch.setattr(StudentV2Target, "train", fake_train)
    monkeypatch.setattr(StudentV2Target, "log_row", _stub_log_row)
    monkeypatch.setattr(StudentV2Target, "METRICS_JSON", fake_repo / "metrics_final_v2.json")

    # Per-role tracker — record which roles got which factory call
    invocations: list[str] = []
    def factory(role: str):
        invocations.append(role)
        # Reuse the canned-response client logic
        return _mock_agent_client_factory()(role)

    tournament.cmd_autoreason(
        "student_v2",
        max_passes=1, convergence=2,
        max_seconds_per_candidate=None,
        hypothesis_seed="test", dry_run=True,
        agent_client_factory=factory,
    )

    # Factory must be called once per role with the role name
    assert sorted(invocations) == ["author", "critic", "synthesizer"]
