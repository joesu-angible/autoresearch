"""Shared adapter base: enforces V2-only writes, V1-file safety, real-run wiring."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
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
    if log_path.name == "results.tsv":
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
    return_code: int = 0


# Default columns matching V1 results.tsv schema, extended with productness keys.
V2_RESULTS_COLUMNS = (
    "round_id", "candidate_id", "candidate_kind",
    "combined_metric", "recall_1", "recall_5", "mean_cosine",
    "productness_pos_acc", "productness_neg_acc",
    "elapsed_seconds", "status", "description",
)


class TargetAdapter:
    """Base class. Subclasses set REPO_DIR, RESULTS_TSV, METRICS_JSON, TRAIN_CMD."""

    name: str = "base"
    REPO_DIR: Path = Path()
    RESULTS_TSV: Path = Path()
    METRICS_JSON: Path = Path()
    TRAIN_CMD: list[str] = []
    DEFAULT_EPOCHS: int = 1

    def apply_patch(self, candidate: Candidate) -> None:
        """No-op for kind='A'; raises if patch touches V1 files.

        Real patch application (writing the diff to the working tree) is the
        responsibility of the orchestration layer — adapters validate, they
        do not mutate the working tree implicitly. Keeps `decide()` reversible.
        """
        if candidate.kind == "A":
            return
        assert_no_v1_writes(candidate.changed_files)

    @property
    def METRICS_PROGRESS_JSON(self) -> Path:
        """Sibling of METRICS_JSON; trainers write this after every eval cycle."""
        return self.METRICS_JSON.parent / "metrics_progress_v2.json"

    SIGTERM_GRACE_SECONDS: float = 30.0

    def train(
        self,
        candidate: Candidate,
        max_epochs: int | None = None,
        dry_run: bool = False,
        log_path: Path | None = None,
        max_seconds: float | None = None,
    ) -> TrainOutcome:
        """Run the training subprocess and return a TrainOutcome.

        Time budget: when `max_seconds` is set, the subprocess is killed
        cleanly (SIGTERM, 30s grace, then SIGKILL) on overrun. Whatever
        progress was written to `metrics_progress_v2.json` up to that point
        is parsed; status becomes "timeout". promote.decide() rejects timeout
        candidates regardless of partial-metric values.
        """
        epochs = max_epochs if max_epochs is not None else self.DEFAULT_EPOCHS
        # Substitute "python" → sys.executable so the trainer subprocess uses
        # the same interpreter as the orchestrator (some servers have no bare
        # `python` on PATH; we should never depend on it).
        cmd = [sys.executable if c == "python" else c for c in self.TRAIN_CMD]
        cmd += ["--max-epochs", str(epochs)]

        if dry_run:
            return TrainOutcome(
                candidate_id=candidate.id,
                metrics={},
                elapsed_seconds=0.0,
                status="noop",
                log_path=log_path or Path("/dev/null"),
                metrics_json_path=self.METRICS_JSON,
                return_code=0,
            )

        log_path = log_path or (self.REPO_DIR / f"run_round_{candidate.round_id}_{candidate.id}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        timed_out = False
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=self.REPO_DIR,
                stdout=logf, stderr=subprocess.STDOUT,
            )
            try:
                proc.wait(timeout=max_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=self.SIGTERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        pass
        elapsed = time.time() - t0

        from research_loop.evaluators import parse_metrics  # local import: avoid cycle
        metrics: dict[str, float] = {}
        status = "failed"
        if timed_out:
            status = "timeout"
            # Recover whatever the trainer wrote up to its last eval (may be empty)
            if self.METRICS_PROGRESS_JSON.exists():
                try:
                    metrics = parse_metrics(self.METRICS_PROGRESS_JSON)
                except Exception:
                    metrics = {}
        elif proc.returncode == 0 and self.METRICS_JSON.exists():
            try:
                metrics = parse_metrics(self.METRICS_JSON)
                status = "success"
            except Exception as e:  # pragma: no cover — diagnostic
                status = f"failed: {e}"
        return TrainOutcome(
            candidate_id=candidate.id,
            metrics=metrics,
            elapsed_seconds=elapsed,
            status=status,
            log_path=log_path,
            metrics_json_path=self.METRICS_JSON,
            return_code=proc.returncode,
        )

    def log_row(self, candidate: Candidate, outcome: TrainOutcome) -> None:
        """Append one row to the V2 results tsv. Refuses V1 paths.

        Auto-creates the tsv with a header row on first write.
        """
        assert_v2_log_path(self.RESULTS_TSV)
        self.RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.RESULTS_TSV.exists() or self.RESULTS_TSV.stat().st_size == 0
        m = outcome.metrics
        row = [
            candidate.round_id,
            candidate.id,
            candidate.kind,
            f"{m.get('combined', 0.0):.6f}",
            f"{m.get('recall_1', 0.0):.4f}",
            f"{m.get('recall_5', 0.0):.4f}",
            f"{m.get('mean_cosine', 0.0):.4f}",
            f"{m.get('productness_pos_acc', 0.0):.4f}",
            f"{m.get('productness_neg_acc', 0.0):.4f}",
            f"{outcome.elapsed_seconds:.1f}",
            outcome.status,
            candidate.hypothesis[:120],
        ]
        with self.RESULTS_TSV.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if new_file:
                writer.writerow(V2_RESULTS_COLUMNS)
            writer.writerow(row)
