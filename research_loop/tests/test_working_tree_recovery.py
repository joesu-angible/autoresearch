"""Tests for resume.recover_working_tree (issue #14 T3)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_loop.candidate import Candidate
from research_loop.resume import (
    WorkingTreeDivergedError,
    recover_working_tree,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Tmp git repo with one tracked file we can apply diffs to."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "foo.py").write_text("ORIGINAL\n")
    subprocess.run(["git", "add", "foo.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _candidate(diff: str, cid: str = "c1", kind: str = "B") -> Candidate:
    return Candidate(
        kind=kind, target="student_v2", round_id="r1",
        hypothesis="x", expected_metric="combined +0.005",
        changed_files=["foo.py"] if kind != "A" else [],
        risks=["x"] if kind != "A" else [],
        rollback="combined < incumbent" if kind != "A" else "N/A",
        patch=diff,
        id=cid,
    )


_PATCH_FOO = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1 +1 @@\n"
    "-ORIGINAL\n"
    "+MODIFIED\n"
)

_PATCH_FOO_2 = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1 +1 @@\n"
    "-ORIGINAL\n"
    "+SECOND_MODIFICATION\n"
)


def _git_clean(repo: Path) -> bool:
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return not res.stdout.strip()


def _apply_patch(repo: Path, patch: str) -> None:
    res = subprocess.run(
        ["git", "apply", "-"], input=patch, cwd=repo,
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_recovers_one_applied_patch(fake_repo: Path):
    _apply_patch(fake_repo, _PATCH_FOO)
    assert not _git_clean(fake_repo)
    cand = _candidate(_PATCH_FOO)
    recover_working_tree(fake_repo, [cand])
    assert _git_clean(fake_repo)
    assert (fake_repo / "foo.py").read_text() == "ORIGINAL\n"


def test_clean_tree_no_op(fake_repo: Path):
    """No applied patch + recovery for a not-applied patch → still clean."""
    cand = _candidate(_PATCH_FOO)
    recover_working_tree(fake_repo, [cand])
    assert _git_clean(fake_repo)


def test_kind_a_empty_patch_no_op(fake_repo: Path):
    """A candidate's empty patch is skipped silently."""
    cand_a = _candidate("", cid="a-id", kind="A")
    recover_working_tree(fake_repo, [cand_a])
    assert _git_clean(fake_repo)


def test_recovers_when_patch_was_never_applied(fake_repo: Path):
    """Mid-apply crash: outcome_started written but apply itself failed.
    Recovery must silently skip (apply -R --check fails) and tree is clean."""
    cand = _candidate(_PATCH_FOO)
    # tree is clean; patch was never applied
    recover_working_tree(fake_repo, [cand])
    assert _git_clean(fake_repo)


# ---------------------------------------------------------------------------
# Diverged-tree detection
# ---------------------------------------------------------------------------

def test_refuses_when_unrelated_file_modified(fake_repo: Path):
    """Operator hand-edited a file not in any unfinished candidate's diff.
    Recovery must refuse and tell them what to do."""
    (fake_repo / "other.py").write_text("HAND EDITED\n")
    subprocess.run(["git", "add", "other.py"], cwd=fake_repo, check=True)
    cand = _candidate(_PATCH_FOO)  # touches foo.py only
    with pytest.raises(WorkingTreeDivergedError, match="other.py"):
        recover_working_tree(fake_repo, [cand])


def test_refuses_when_apply_fails_and_tree_still_dirty(fake_repo: Path):
    """Patch context doesn't match (file was hand-edited differently) →
    apply -R fails AND tree stays dirty → must raise."""
    # Operator changed foo.py to something the patch doesn't recognize
    (fake_repo / "foo.py").write_text("UNRELATED CHANGE\n")
    cand = _candidate(_PATCH_FOO)  # expects to revert MODIFIED→ORIGINAL
    with pytest.raises(WorkingTreeDivergedError):
        recover_working_tree(fake_repo, [cand])
