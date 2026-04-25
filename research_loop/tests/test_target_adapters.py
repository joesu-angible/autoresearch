"""T12-T13 verification: target adapters refuse V1 writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_loop.candidate import Candidate
from research_loop.targets import DinoV2Target, StudentV2Target
from research_loop.targets._base import (
    V1_FORBIDDEN_PATHS,
    assert_no_v1_writes,
    assert_v2_log_path,
)


def _b(target: str, changed: list[str]):
    return Candidate(
        kind="B",
        target=target,
        hypothesis="add stronger augmentation to V2 trainer",
        expected_metric="combined +0.005",
        changed_files=changed,
        risks=["may slow training"],
        rollback="combined < 0.855",
        patch="--- a\n+++ b\n",
    )


def test_v1_forbidden_paths_listed():
    assert "student_finetune/train.py" in V1_FORBIDDEN_PATHS
    assert "student_finetune/train_final.py" in V1_FORBIDDEN_PATHS
    assert "student_finetune/prepare.py" in V1_FORBIDDEN_PATHS
    assert "student_finetune/results.tsv" in V1_FORBIDDEN_PATHS
    assert "dino_finetune/train_dino.py" in V1_FORBIDDEN_PATHS


def test_assert_no_v1_writes_passes_for_v2_only():
    assert_no_v1_writes(["student_finetune/train_v2.py"])
    assert_no_v1_writes(["dino_finetune/train_dino_v2.py"])


@pytest.mark.parametrize("forbidden", V1_FORBIDDEN_PATHS)
def test_assert_no_v1_writes_rejects_each_forbidden(forbidden: str):
    with pytest.raises(ValueError, match="Tournament cannot modify V1 file"):
        assert_no_v1_writes([forbidden])


def test_assert_v2_log_path_accepts_results_v2():
    assert_v2_log_path(Path("student_finetune/results_v2.tsv"))
    assert_v2_log_path(Path("dino_finetune/results_v2.tsv"))


def test_assert_v2_log_path_rejects_v1_results():
    with pytest.raises(ValueError):
        assert_v2_log_path(Path("student_finetune/results.tsv"))


def test_student_v2_apply_patch_a_is_noop():
    a = Candidate(
        kind="A", target="student_v2", hypothesis="incumbent",
        expected_metric="unchanged", changed_files=[], risks=[],
        rollback="N/A", patch="",
    )
    StudentV2Target().apply_patch(a)  # must not raise


def test_student_v2_apply_patch_rejects_v1_file():
    bad = _b("student_v2", changed=["student_finetune/train.py"])
    with pytest.raises(ValueError):
        StudentV2Target().apply_patch(bad)


def test_dino_v2_apply_patch_rejects_v1_file():
    bad = _b("dino_v2", changed=["dino_finetune/train_dino.py"])
    with pytest.raises(ValueError):
        DinoV2Target().apply_patch(bad)


def test_student_v2_results_path_is_v2():
    s = StudentV2Target()
    assert s.RESULTS_TSV.name == "results_v2.tsv"
    assert "student_finetune" in str(s.RESULTS_TSV)


def test_dino_v2_results_path_is_v2():
    d = DinoV2Target()
    assert d.RESULTS_TSV.name == "results_v2.tsv"
    assert "dino_finetune" in str(d.RESULTS_TSV)


def test_train_dry_run_does_not_execute():
    """dry_run=True must return without launching any subprocess."""
    rc, log = StudentV2Target().train(max_epochs=1, dry_run=True)
    assert rc == 0
    assert "[dry-run]" in log
