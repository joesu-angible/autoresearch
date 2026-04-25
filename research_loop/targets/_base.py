"""Shared adapter base: enforces V2-only writes and V1-file safety."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from research_loop.candidate import Candidate

# Paths the tournament must never write to (V1 history is read-only).
V1_FORBIDDEN_PATHS: tuple[str, ...] = (
    "student_finetune/results.tsv",
    "dino_finetune/results.tsv",
    "student_finetune/train.py",
    "student_finetune/train_final.py",
    "dino_finetune/train_dino.py",
    "student_finetune/prepare.py",
)


def assert_no_v1_writes(changed_files: list[str]) -> None:
    """Raise ValueError if any path matches a V1-forbidden file."""
    for cf in changed_files:
        for forbidden in V1_FORBIDDEN_PATHS:
            if cf.endswith(forbidden) or cf == forbidden:
                raise ValueError(
                    f"Tournament cannot modify V1 file: {cf} "
                    f"(matched {forbidden})"
                )


def assert_v2_log_path(log_path: Path) -> None:
    if log_path.name in ("results.tsv",) or "results.tsv" == str(log_path).split("/")[-1]:
        raise ValueError(f"Refusing to write into V1 log path: {log_path}")
    if not log_path.name.endswith("_v2.tsv") and "results_v2" not in log_path.name:
        raise ValueError(f"Tournament adapter must log to a *_v2.tsv path; got: {log_path}")


@dataclass
class TrainOutcome:
    candidate_id: str
    metrics: dict[str, float]
    elapsed_seconds: float
    status: str  # "success" | "failed" | "noop"
    log_path: Path
    metrics_json_path: Path


class TargetAdapter:
    """Base class. Subclasses set REPO_DIR, RESULTS_TSV, METRICS_JSON, TRAIN_CMD."""

    name: str = "base"
    REPO_DIR: Path = Path()
    RESULTS_TSV: Path = Path()
    METRICS_JSON: Path = Path()
    TRAIN_CMD: list[str] = []

    def apply_patch(self, candidate: Candidate) -> None:
        """No-op for kind='A'; raises if patch touches V1 files."""
        if candidate.kind == "A":
            return
        assert_no_v1_writes(candidate.changed_files)
        # Real patch application is delegated to the tournament harness; this
        # check makes "would this be safe to apply" testable without touching
        # working tree.

    def train(self, max_epochs: int = 1, dry_run: bool = False) -> tuple[int, str]:
        """Run the training subprocess. Returns (returncode, log_text)."""
        cmd = list(self.TRAIN_CMD) + ["--max-epochs", str(max_epochs)]
        if dry_run:
            return 0, f"[dry-run] would exec: {cmd}"
        proc = subprocess.run(
            cmd, cwd=self.REPO_DIR, capture_output=True, text=True, check=False
        )
        return proc.returncode, (proc.stdout + proc.stderr)

    def log_row(self, row: list[str]) -> None:
        """Append one row to the V2 results tsv. Refuses V1 paths."""
        assert_v2_log_path(self.RESULTS_TSV)
        self.RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
        with self.RESULTS_TSV.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)
