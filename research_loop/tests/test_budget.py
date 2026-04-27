"""T2 verification: per-trial time budget on adapter.train().

Uses a stub trainer (a tiny shell script written into a tmp dir) instead of
the real GPU pipeline. Asserts:
  - subprocess killed within 60s of max_seconds expiry (we use 1-3s in tests)
  - status == "timeout"
  - partial metrics from metrics_progress_v2.json are recovered when present
  - promote.decide() rejects timeout candidates regardless of metric values
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from research_loop.candidate import Candidate
from research_loop.promote import CandidateResult, decide
from research_loop.targets._base import TargetAdapter, TrainOutcome


@pytest.fixture
def stub_trainer_factory(tmp_path: Path):
    """Build a stub trainer dir that simulates a slow Python trainer.

    The trainer writes metrics_progress_v2.json after a short delay, then
    sleeps for `sleep_seconds` (so the adapter's max_seconds budget kicks in).
    Returns the adapter pointed at this stub.
    """

    def _make(sleep_seconds: float = 5.0, write_progress: bool = True):
        trainer = tmp_path / "fake_train.py"
        trainer.write_text(
            "import json, sys, time\n"
            "from pathlib import Path\n"
            "out = Path(__file__).parent / 'metrics_progress_v2.json'\n"
            f"if {write_progress}:\n"
            "    payload = {\n"
            "        'status': 'in_progress', 'is_partial': True,\n"
            "        'epochs_completed': 1, 'max_epochs': 30,\n"
            "        'combined_metric': 0.42, 'best_combined': 0.42,\n"
            "        'recall_at_1': 0.5, 'recall_at_5': 0.6,\n"
            "        'mean_cosine': 0.34, 'distill_loss': 1.2,\n"
            "    }\n"
            "    out.write_text(json.dumps(payload))\n"
            "time.sleep(0.2)  # let adapter see the progress file\n"
            f"time.sleep({sleep_seconds})\n"
        )

        class StubAdapter(TargetAdapter):
            name = "stub_v2"
            REPO_DIR = tmp_path
            RESULTS_TSV = tmp_path / "results_v2.tsv"
            METRICS_JSON = tmp_path / "metrics_final_v2.json"
            TRAIN_CMD = ["python", str(trainer)]
            DEFAULT_EPOCHS = 1
            SIGTERM_GRACE_SECONDS = 1.0  # speed up tests

        return StubAdapter()

    return _make


def _candidate():
    return Candidate(
        kind="A", target="student_v2", round_id="r-budget-test",
        hypothesis="incumbent baseline",
        expected_metric="combined Δ +0.000", changed_files=[], risks=[],
        rollback="N/A", patch="",
    )


def test_budget_kills_overrunning_subprocess(stub_trainer_factory, monkeypatch):
    """Adapter keeps a wall-clock watchdog for trainers that ignore the env budget."""
    monkeypatch.setenv("AUTORESEARCH_CANDIDATE_WALL_GRACE_SECONDS", "1")
    adapter = stub_trainer_factory(sleep_seconds=30, write_progress=False)
    candidate = _candidate()
    t0 = time.time()
    outcome = adapter.train(candidate, max_epochs=1, max_seconds=3.0)
    elapsed = time.time() - t0
    assert outcome.status == "timeout"
    # Killed shortly after budget + 1s grace + adapter SIGTERM grace + small slack
    assert elapsed < 10.0, f"budget overrun: elapsed={elapsed:.1f}s"


def test_adapter_passes_full_training_budget_env_and_larger_wall_timeout(tmp_path, monkeypatch):
    """max_seconds is the trainer's own train-step budget, not the whole subprocess wall time."""
    monkeypatch.setenv("AUTORESEARCH_CANDIDATE_WALL_GRACE_SECONDS", "2")
    trainer = tmp_path / "fake_train.py"
    trainer.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "budget = os.environ.get('MAX_TRAINING_SECONDS')\n"
        "time.sleep(1.2)  # simulate setup outside training budget\n"
        "out = Path(__file__).parent / 'metrics_final_v2.json'\n"
        "out.write_text(json.dumps({'combined_metric': 0.7, 'recall_at_1': 0.6, 'mean_cosine': 0.5, 'budget': budget}))\n"
    )

    class StubAdapter(TargetAdapter):
        name = "stub_v2"
        REPO_DIR = tmp_path
        RESULTS_TSV = tmp_path / "results_v2.tsv"
        METRICS_JSON = tmp_path / "metrics_final_v2.json"
        TRAIN_CMD = ["python", str(trainer)]
        DEFAULT_EPOCHS = 1

    outcome = StubAdapter().train(_candidate(), max_epochs=1, max_seconds=1.0)
    assert outcome.status == "success"
    assert json.loads((tmp_path / "metrics_final_v2.json").read_text())["budget"] == "1"


def test_budget_recovers_partial_metrics_from_progress_json(stub_trainer_factory, monkeypatch):
    """Stub writes metrics_progress; adapter parses on watchdog timeout."""
    monkeypatch.setenv("AUTORESEARCH_CANDIDATE_WALL_GRACE_SECONDS", "1")
    adapter = stub_trainer_factory(sleep_seconds=30, write_progress=True)
    outcome = adapter.train(_candidate(), max_epochs=1, max_seconds=3.0)
    assert outcome.status == "timeout"
    # Progress was written before the sleep, so adapter should have it
    assert outcome.metrics
    assert outcome.metrics.get("combined") == 0.42
    assert outcome.metrics.get("recall_1") == 0.5


def test_budget_no_progress_file_yields_empty_metrics(stub_trainer_factory, monkeypatch):
    """Watchdog timeout before any eval → no progress file → empty metrics dict."""
    monkeypatch.setenv("AUTORESEARCH_CANDIDATE_WALL_GRACE_SECONDS", "1")
    adapter = stub_trainer_factory(sleep_seconds=30, write_progress=False)
    outcome = adapter.train(_candidate(), max_epochs=1, max_seconds=2.0)
    assert outcome.status == "timeout"
    assert outcome.metrics == {}


def test_no_budget_means_no_timeout(stub_trainer_factory):
    """When max_seconds=None the adapter waits indefinitely (current behavior preserved)."""
    # Use sleep=0.5 so the test stays fast
    adapter = stub_trainer_factory(sleep_seconds=0.5, write_progress=False)
    outcome = adapter.train(_candidate(), max_epochs=1, max_seconds=None)
    # Trainer didn't write metrics_final_v2.json so this counts as "failed"
    # (the trainer-success path is exercised by test_tournament_flow.py).
    assert outcome.status != "timeout"


def test_promote_decide_rejects_timeout_outcome():
    """A timeout candidate cannot win, regardless of how good the partial metrics look."""
    a = CandidateResult(
        candidate_id="a", kind="A",
        combined=0.86, recall_1=0.90, mean_cosine=0.81,
        productness_pos_acc=0.99, productness_neg_acc=0.85,
        status="success",
    )
    b_timeout = CandidateResult(
        candidate_id="b", kind="B",
        # Partial metrics happen to look better than A — must NOT win
        combined=0.95, recall_1=0.95, mean_cosine=0.90,
        productness_pos_acc=0.99, productness_neg_acc=0.90,
        status="timeout",  # ← the veto
    )
    decision = decide([a, b_timeout])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_promote_decide_rejects_failed_outcome():
    """Same veto applies to status='failed'."""
    a = CandidateResult(
        candidate_id="a", kind="A",
        combined=0.86, recall_1=0.90, mean_cosine=0.81,
        status="success",
    )
    b_failed = CandidateResult(
        candidate_id="b", kind="B",
        combined=0.99, recall_1=0.99, mean_cosine=0.99,
        status="failed: import error",
    )
    decision = decide([a, b_failed])
    assert decision.winner_kind == "A"
