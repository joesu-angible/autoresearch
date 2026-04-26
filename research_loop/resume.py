"""Resume primitives for autoreason runs (issue #14).

A crashed autoreason run leaves enough state on disk for resume to pick up:

  research_loop/runs/<run_id>/
    summary.json     ← target, max_passes, convergence, LLM config, ...
    autoreason.pid   ← prior runner's PID
    autoreason.log   ← narrative log (appended to on resume)

  research_loop/history.jsonl
    ... outcome_started records mark candidates that began training
    ... matching outcome records mark completion (success/failed/timeout)

This module exposes the read-side primitives and the working-tree recovery
helper that `cmd_autoreason --resume` composes into a full resume flow.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from research_loop.candidate import Candidate, read_decisions, read_outcomes_started


class WorkingTreeDivergedError(RuntimeError):
    """Working tree has modifications that resume cannot safely revert."""


def find_run_dir(runs_dir: Path, run_id: str) -> Path | None:
    """Return runs_dir/<run_id> iff it exists; None otherwise."""
    candidate = runs_dir / run_id
    return candidate if candidate.is_dir() else None


def check_pid_dead(run_dir: Path) -> bool:
    """True if the prior runner is dead (or its PID file is missing).

    A live PID means another runner is already working on this run — refuse
    resume in that case. A missing PID file is also fine: the prior runner
    was killed before it could write the file, or this is a fresh run dir.
    """
    pid_file = run_dir / "autoreason.pid"
    if not pid_file.exists():
        return True
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return True
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)  # signal 0 = liveness probe; raises if dead
    except ProcessLookupError:
        return True
    except PermissionError:
        # Process exists but we can't signal it (different uid). Treat as alive.
        return False
    return False


def load_run_config(run_dir: Path) -> dict:
    """Read summary.json and return the persisted run config block.

    Backward compatible: a summary.json from before T2 has no `config` key
    and this returns {}; the caller is responsible for falling back to its
    own defaults / explicit CLI flags in that case.
    """
    summary = run_dir / "summary.json"
    if not summary.exists():
        return {}
    try:
        data = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return {}
    cfg = data.get("config")
    return dict(cfg) if isinstance(cfg, dict) else {}


def load_run_target(run_dir: Path) -> str | None:
    """Convenience: read the target from summary.json (top-level field)."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return None
    return data.get("target")


def round_ids_for_run(history_path: Path, run_id: str) -> set[str]:
    """Round_ids belonging to a single autoreason run.

    Decisions don't carry run_id directly; we infer it via outcome_started
    records (which do carry run_id) → round_id mapping. A round whose pass
    crashed before any candidate began training is excluded — but no decision
    will exist for it either, so the streak math is unaffected.
    """
    return {s.round_id for s in read_outcomes_started(history_path, run_id=run_id)}


def compute_consecutive_a_wins(
    history_path: Path, target: str, convergence: int,
    run_id: str | None = None,
) -> int:
    """Count trailing A wins in this target's decision stream (capped at convergence).

    Walks decisions newest-first and counts a streak of `winner_kind="A"`
    until a non-A decision (or the start of history) is reached. Capped at
    `convergence` so resume cannot trigger convergence twice with the same
    decisions.

    When `run_id` is provided, the streak is scoped to decisions that belong
    to this run (via outcome_started → round_id mapping). Without this scope,
    a target's decisions from prior runs would pollute the counter — a fresh
    resume of a target that previously converged would see "consecutive_a_wins
    = convergence" immediately and exit without running anything.
    """
    decisions = [d for d in read_decisions(history_path) if d.target == target]
    if run_id is not None:
        rids = round_ids_for_run(history_path, run_id)
        decisions = [d for d in decisions if d.round_id in rids]
    if not decisions:
        return 0
    streak = 0
    for d in reversed(decisions):
        if d.winner_kind == "A":
            streak += 1
            if streak >= convergence:
                return convergence
        else:
            break
    return streak


def count_decisions_for_run(
    history_path: Path, target: str, run_id: str,
) -> int:
    """Number of completed decisions belonging to this run (target-scoped)."""
    rids = round_ids_for_run(history_path, run_id)
    return sum(
        1 for d in read_decisions(history_path)
        if d.target == target and d.round_id in rids
    )


# ---------------------------------------------------------------------------
# Working-tree recovery (T3)
# ---------------------------------------------------------------------------

def _git_status_porcelain(repo: Path) -> list[str]:
    """Return `git status --porcelain` lines (empty list = clean tree)."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git status failed: {res.stderr.strip()}")
    return [line for line in res.stdout.splitlines() if line.strip()]


def _git_apply_reverse(repo: Path, diff_text: str) -> bool:
    """Try `git apply -R` of `diff_text` against `repo`. Return True on success.

    Uses `--check` first so a not-applied patch fails cleanly without
    mutating the working tree.
    """
    if not diff_text.strip():
        return True  # empty patch = nothing to revert
    check = subprocess.run(
        ["git", "apply", "--check", "-R", "-"],
        input=diff_text, cwd=repo, capture_output=True, text=True, check=False,
    )
    if check.returncode != 0:
        return False
    apply_res = subprocess.run(
        ["git", "apply", "-R", "-"],
        input=diff_text, cwd=repo, capture_output=True, text=True, check=False,
    )
    return apply_res.returncode == 0


def _files_in_diff(diff_text: str) -> set[str]:
    """Repo-relative paths touched by a unified diff."""
    paths: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            paths.add(line.split("/", 1)[1].strip())
    paths.discard("_noop")  # autoreason placeholder
    return paths


def recover_working_tree(repo: Path, unfinished: list[Candidate]) -> None:
    """Best-effort revert of unfinished candidates' patches; refuse on diverged tree.

    1. For each unfinished candidate's non-empty patch: try `git apply -R`.
       Silently skip patches that aren't actually applied.
    2. After all attempts, `git status --porcelain` must be empty. If not,
       check whether the dirty paths are explainable by an unfinished
       candidate's diff. If yes, raise WorkingTreeDivergedError (we tried to
       revert and failed). If no, raise WorkingTreeDivergedError listing the
       unexplained paths — the operator hand-edited or pulled.
    """
    for cand in unfinished:
        if cand.patch.strip():
            _git_apply_reverse(repo, cand.patch)

    dirty = _git_status_porcelain(repo)
    if not dirty:
        return

    explained_paths: set[str] = set()
    for cand in unfinished:
        explained_paths |= _files_in_diff(cand.patch)

    unexplained: list[str] = []
    for line in dirty:
        # Porcelain format: "XY path" — XY are status codes, path may have leading space
        path = line[3:].strip()
        # Strip "renamed-from -> renamed-to" form
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in explained_paths:
            unexplained.append(path)

    if unexplained:
        raise WorkingTreeDivergedError(
            f"Working tree has modifications outside the unfinished candidates' "
            f"diffs: {unexplained}. Commit, stash, or revert them before resuming."
        )
    raise WorkingTreeDivergedError(
        f"Could not fully revert unfinished candidates' patches; tree still "
        f"dirty in: {[line[3:].strip() for line in dirty]}. Manual cleanup required."
    )
