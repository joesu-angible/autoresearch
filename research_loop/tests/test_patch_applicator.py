"""T3 verification: research_loop.patch.apply_patch().

Spins up tmp git repos as fixtures (no monkeypatch — real git invocations).
Tests the contract: apply mutates, revert restores; V1 paths refused before
any mutation; malformed diffs raise; revert runs even when the body raises.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research_loop.patch import apply_patch, diff_touches_v1


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A clean tmp git repo with one tracked file resembling student_finetune/train_v2.py."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)

    (tmp_path / "student_finetune").mkdir()
    target = tmp_path / "student_finetune" / "train_v2.py"
    target.write_text(
        "USE_PRODUCTNESS_CLS = True\n"
        "PRODUCTNESS_CLS_WEIGHT = 0.02\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


def _make_diff(target_rel: str, old: str, new: str) -> str:
    """Build a minimal unified diff (one-line replace, sufficient for apply)."""
    return (
        f"diff --git a/{target_rel} b/{target_rel}\n"
        f"--- a/{target_rel}\n"
        f"+++ b/{target_rel}\n"
        f"@@ -1,2 +1,2 @@\n"
        f"-{old}\n"
        f"+{new}\n"
        f" PRODUCTNESS_CLS_WEIGHT = 0.02\n"
    )


# ---------------------------------------------------------------------------
# diff_touches_v1
# ---------------------------------------------------------------------------

def test_diff_touches_v1_detects_train_py():
    diff = "--- a/student_finetune/train.py\n+++ b/student_finetune/train.py\n"
    assert "student_finetune/train.py" in diff_touches_v1(diff)


def test_diff_touches_v1_passes_v2_files():
    diff = "--- a/student_finetune/train_v2.py\n+++ b/student_finetune/train_v2.py\n"
    assert diff_touches_v1(diff) == []


def test_diff_touches_v1_detects_results_tsv():
    diff = "--- a/student_finetune/results.tsv\n+++ b/student_finetune/results.tsv\n"
    assert "student_finetune/results.tsv" in diff_touches_v1(diff)


def test_diff_touches_v1_detects_prepare_py():
    diff = (
        "diff --git a/student_finetune/prepare.py b/student_finetune/prepare.py\n"
        "--- a/student_finetune/prepare.py\n"
        "+++ b/student_finetune/prepare.py\n"
    )
    assert "student_finetune/prepare.py" in diff_touches_v1(diff)


# ---------------------------------------------------------------------------
# apply_patch contract
# ---------------------------------------------------------------------------

def test_apply_then_revert_restores_tree(git_repo: Path):
    target = git_repo / "student_finetune" / "train_v2.py"
    original = target.read_text()
    diff = _make_diff(
        "student_finetune/train_v2.py",
        "USE_PRODUCTNESS_CLS = True",
        "USE_PRODUCTNESS_CLS = False",
    )
    with apply_patch(diff, repo=git_repo):
        assert "USE_PRODUCTNESS_CLS = False" in target.read_text()
    # After context exit: original restored
    assert target.read_text() == original


def test_revert_runs_even_when_body_raises(git_repo: Path):
    target = git_repo / "student_finetune" / "train_v2.py"
    original = target.read_text()
    diff = _make_diff(
        "student_finetune/train_v2.py",
        "USE_PRODUCTNESS_CLS = True",
        "USE_PRODUCTNESS_CLS = False",
    )
    with pytest.raises(RuntimeError, match="boom"):
        with apply_patch(diff, repo=git_repo):
            assert "USE_PRODUCTNESS_CLS = False" in target.read_text()
            raise RuntimeError("boom")
    # Body raised, but revert still ran:
    assert target.read_text() == original


def test_apply_refuses_v1_path_before_mutation(git_repo: Path):
    # Add a V1 file so we can verify it stays untouched
    v1 = git_repo / "student_finetune" / "train.py"
    v1.write_text("# V1 sacred\n")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "v1"], cwd=git_repo, check=True)

    diff = _make_diff(
        "student_finetune/train.py",
        "# V1 sacred",
        "# V1 violated",
    )
    original = v1.read_text()
    with pytest.raises(ValueError, match="V1_FORBIDDEN_PATHS"):
        with apply_patch(diff, repo=git_repo):
            pass
    assert v1.read_text() == original  # not touched


def test_malformed_diff_raises_called_process_error(git_repo: Path):
    bad_diff = "this is not a diff\n"
    # Empty by parser standards (no +++/--- headers) → counted as no-op, yields cleanly.
    # Provide something that *looks* like a diff but won't apply.
    bad_diff = (
        "--- a/student_finetune/train_v2.py\n"
        "+++ b/student_finetune/train_v2.py\n"
        "@@ -99,2 +99,2 @@\n"  # nonexistent line range
        "-IMPOSSIBLE_LINE\n"
        "+IMPOSSIBLE_REPLACE\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        with apply_patch(bad_diff, repo=git_repo):
            pass


def test_empty_diff_is_noop(git_repo: Path):
    """kind='A' baseline has empty patch — must yield without mutation."""
    target = git_repo / "student_finetune" / "train_v2.py"
    original = target.read_text()
    with apply_patch("", repo=git_repo):
        assert target.read_text() == original


def test_dirty_tree_refuses_apply(git_repo: Path):
    target = git_repo / "student_finetune" / "train_v2.py"
    target.write_text(target.read_text() + "\n# dirty\n")  # uncommitted change

    diff = _make_diff(
        "student_finetune/train_v2.py",
        "USE_PRODUCTNESS_CLS = True",
        "USE_PRODUCTNESS_CLS = False",
    )
    with pytest.raises(RuntimeError, match="Working tree is dirty"):
        with apply_patch(diff, repo=git_repo):
            pass


def test_dirty_tree_check_can_be_overridden(git_repo: Path):
    """Tests that exercise a temp repo built fresh per-test should be allowed
    to skip the cleanliness check via require_clean_tree=False."""
    diff = _make_diff(
        "student_finetune/train_v2.py",
        "USE_PRODUCTNESS_CLS = True",
        "USE_PRODUCTNESS_CLS = False",
    )
    # Add a stray untracked file to make the tree dirty
    (git_repo / "stray.txt").write_text("scratch")
    with apply_patch(diff, repo=git_repo, require_clean_tree=False):
        assert (git_repo / "student_finetune" / "train_v2.py").read_text().count("False") == 1
