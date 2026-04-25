"""Tests for research_loop.resume primitives (issue #14 T2)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from research_loop.candidate import Decision, append_history
from research_loop.resume import (
    check_pid_dead,
    compute_consecutive_a_wins,
    find_run_dir,
    load_run_config,
    load_run_target,
)


# ---------------------------------------------------------------------------
# find_run_dir
# ---------------------------------------------------------------------------

def test_find_run_dir_returns_path_when_present(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    rd = runs_dir / "run-abc"
    rd.mkdir(parents=True)
    assert find_run_dir(runs_dir, "run-abc") == rd


def test_find_run_dir_returns_none_when_missing(tmp_path: Path):
    assert find_run_dir(tmp_path / "runs", "no-such-run") is None


def test_find_run_dir_returns_none_for_file_not_dir(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "run-abc").write_text("not a dir")
    assert find_run_dir(runs_dir, "run-abc") is None


# ---------------------------------------------------------------------------
# check_pid_dead
# ---------------------------------------------------------------------------

def test_check_pid_dead_missing_pid_file(tmp_path: Path):
    """No PID file → assume dead (fresh run dir or runner killed before write)."""
    assert check_pid_dead(tmp_path) is True


def test_check_pid_dead_with_alive_pid(tmp_path: Path):
    """Our own PID is definitely alive."""
    (tmp_path / "autoreason.pid").write_text(str(os.getpid()))
    assert check_pid_dead(tmp_path) is False


def test_check_pid_dead_with_dead_pid(tmp_path: Path):
    """Spawn a short subprocess, wait for it to die, check its PID."""
    proc = subprocess.Popen(["true"])
    proc.wait()  # zombie reaped on wait()
    # Re-check after a beat to ensure the kernel has fully released the PID
    time.sleep(0.05)
    (tmp_path / "autoreason.pid").write_text(str(proc.pid))
    # Note: the test relies on PID not being immediately recycled. In practice
    # this is reliable on Linux for short-lived test runs.
    assert check_pid_dead(tmp_path) is True


def test_check_pid_dead_malformed_pid_file(tmp_path: Path):
    (tmp_path / "autoreason.pid").write_text("not-a-number")
    assert check_pid_dead(tmp_path) is True


def test_check_pid_dead_empty_pid_file(tmp_path: Path):
    (tmp_path / "autoreason.pid").write_text("")
    assert check_pid_dead(tmp_path) is True


# ---------------------------------------------------------------------------
# load_run_config / load_run_target
# ---------------------------------------------------------------------------

def test_load_run_config_round_trip(tmp_path: Path):
    cfg = {
        "target": "student_v2",
        "max_passes": 15,
        "convergence": 2,
        "llm_cli": "hermes",
        "llm_model": "openai-codex/gpt-5-codex",
    }
    summary = {"run_id": "r1", "target": "student_v2", "config": cfg}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert load_run_config(tmp_path) == cfg


def test_load_run_config_backward_compat_no_config_key(tmp_path: Path):
    """summary.json from before T2 has no `config` field → returns {}."""
    summary = {"run_id": "r1", "target": "student_v2"}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert load_run_config(tmp_path) == {}


def test_load_run_config_missing_summary(tmp_path: Path):
    assert load_run_config(tmp_path) == {}


def test_load_run_config_malformed_summary(tmp_path: Path):
    (tmp_path / "summary.json").write_text("not json {")
    assert load_run_config(tmp_path) == {}


def test_load_run_target_returns_target(tmp_path: Path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "run_id": "r1", "target": "dino_v2", "config": {},
    }))
    assert load_run_target(tmp_path) == "dino_v2"


def test_load_run_target_missing_summary(tmp_path: Path):
    assert load_run_target(tmp_path) is None


# ---------------------------------------------------------------------------
# compute_consecutive_a_wins
# ---------------------------------------------------------------------------

def _decision(round_id: str, kind: str, target: str = "student_v2") -> Decision:
    return Decision(
        round_id=round_id, target=target,
        winner_id=f"id-{round_id}", winner_kind=kind,
        promote=(kind != "A"), reason="",
        deployable=False, deploy_failures=(),
    )


def test_consecutive_a_wins_empty_history(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    assert compute_consecutive_a_wins(h, "student_v2", convergence=2) == 0


def test_consecutive_a_wins_streak_at_end(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    append_history(h, _decision("r1", "B"))
    append_history(h, _decision("r2", "A"))
    append_history(h, _decision("r3", "A"))  # 2-streak
    assert compute_consecutive_a_wins(h, "student_v2", convergence=2) == 2


def test_consecutive_a_wins_resets_on_b(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    append_history(h, _decision("r1", "A"))
    append_history(h, _decision("r2", "A"))
    append_history(h, _decision("r3", "B"))  # streak broken
    append_history(h, _decision("r4", "A"))
    assert compute_consecutive_a_wins(h, "student_v2", convergence=2) == 1


def test_consecutive_a_wins_caps_at_convergence(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    for i in range(5):
        append_history(h, _decision(f"r{i}", "A"))
    # 5 A wins, but cap at convergence=2
    assert compute_consecutive_a_wins(h, "student_v2", convergence=2) == 2


def test_consecutive_a_wins_filters_by_target(tmp_path: Path):
    h = tmp_path / "history.jsonl"
    append_history(h, _decision("r1", "A", target="dino_v2"))
    append_history(h, _decision("r2", "A", target="student_v2"))
    append_history(h, _decision("r3", "A", target="dino_v2"))
    # Student stream only has 1 A
    assert compute_consecutive_a_wins(h, "student_v2", convergence=5) == 1
    # DINO stream has 2 (interspersed with student doesn't break the streak)
    assert compute_consecutive_a_wins(h, "dino_v2", convergence=5) == 2
