"""Worktree-based authoring helpers.

These helpers are the "real edit" path for autoreason candidates: instead of
asking an LLM to hand-write unified diffs, run the author/synthesizer inside an
isolated git worktree, let it edit files, and then ask git to produce the diff.
That matches the original manual autoresearch loops documented in program_*.md:
the agent changes files, then git records the resulting patch.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from research_loop.agents.author import AUTHOR_B_SYSTEM_PROMPT, PatchProposal
from research_loop.agents.synthesizer import SYNTHESIZER_SYSTEM_PROMPT
from research_loop.targets._base import V1_FORBIDDEN_PATHS


class FileEditingClient(Protocol):
    name: str

    def edit_files(
        self,
        system: str,
        user: str,
        *,
        workdir: Path,
        timeout: float | None = None,
    ) -> str:
        """Run an agent in workdir, allowing it to edit files; return final text."""


def _run_git(repo: Path, args: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=repo,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_clean(repo_root: Path) -> None:
    status = _run_git(repo_root, ["status", "--porcelain"])
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr}")
    if status.stdout.strip():
        raise RuntimeError(
            "Cannot create author worktree from dirty repo; commit/stash first.\n"
            f"git status:\n{status.stdout}"
        )


def _extract_rationale(raw: str) -> str:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "No final rationale returned by editing agent."
    # Keep this short because PatchProposalRecord stores raw separately.
    return " ".join(lines)[:800]


def _changed_files(worktree: Path) -> list[str]:
    status = _run_git(worktree, ["status", "--porcelain"])
    if status.returncode != 0:
        raise RuntimeError(f"git status failed in worktree: {status.stderr}")
    files: list[str] = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        # Porcelain v1 path starts at column 4 for normal entries. Renames are
        # not expected here, but keep the destination side if present.
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


def _validate_changed_files(files: list[str], allowed_paths: set[str]) -> None:
    illegal = sorted(set(files) - allowed_paths)
    forbidden = []
    for path in files:
        for protected in V1_FORBIDDEN_PATHS:
            if path == protected or path.endswith("/" + protected) or path.endswith(protected):
                forbidden.append(path)
                break
    if illegal or forbidden:
        details = []
        if illegal:
            details.append(f"unexpected files: {illegal}")
        if forbidden:
            details.append(f"V1-protected files: {forbidden}")
        raise RuntimeError("Editing agent modified disallowed files (" + "; ".join(details) + ")")


def _git_diff(worktree: Path, paths: list[str]) -> str:
    diff = _run_git(worktree, ["diff", "--", *paths])
    if diff.returncode != 0:
        raise RuntimeError(f"git diff failed in worktree: {diff.stderr}")
    return diff.stdout


def _with_git_worktree(repo_root: Path, fn):
    tmp_parent = Path(tempfile.mkdtemp(prefix="autoreason-author-"))
    worktree = tmp_parent / "worktree"
    try:
        add = _run_git(repo_root, ["worktree", "add", "--detach", str(worktree), "HEAD"])
        if add.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {add.stderr}")
        return fn(worktree)
    finally:
        # `git worktree remove` cleans up metadata; shutil handles partially
        # created directories if add failed before registering the worktree.
        if worktree.exists():
            _run_git(repo_root, ["worktree", "remove", "--force", str(worktree)])
        shutil.rmtree(tmp_parent, ignore_errors=True)


def author_diff_in_worktree(
    client: FileEditingClient,
    *,
    repo_root: Path,
    trainer_path: str,
    trainer_source: str,
    critique_text: str,
    timeout: float | None = None,
) -> PatchProposal:
    """Ask an editing agent to change trainer_path; return git-generated diff."""

    def run(worktree: Path) -> PatchProposal:
        user = f"""
You are editing a real git worktree. Do NOT output a unified diff by hand.

Task:
- Modify ONLY `{trainer_path}`.
- Address the critic findings conservatively with the smallest useful change.
- Preserve all V1-protected files.
- Prior failed/timeout outcomes are orchestration evidence, not proof that the trainer has no useful ML experiment left; do not let old execution failures alone justify a no-op.
- When the critic identifies actionable trainer issues, generate at least one concrete safe trainer experiment. Prefer a tiny hyperparameter/guardrail/checkpoint-state change over no-op. Leave files unchanged only when there is truly no safe trainer-side experiment.

After editing, your final response should be only a short rationale paragraph.
Autoreason will run `git diff` itself to produce the candidate patch.

## Critic findings
{critique_text}

## Current `{trainer_path}` source
```python
{trainer_source}
```
""".strip()
        raw = client.edit_files(AUTHOR_B_SYSTEM_PROMPT, user, workdir=worktree, timeout=timeout)
        files = _changed_files(worktree)
        _validate_changed_files(files, {trainer_path})
        diff = _git_diff(worktree, [trainer_path]) if files else ""
        return PatchProposal(
            rationale=_extract_rationale(raw),
            diff=diff,
            raw=raw,
        )

    return _with_git_worktree(repo_root, run)


def synthesize_diff_in_worktree(
    client: FileEditingClient,
    *,
    repo_root: Path,
    trainer_path: str,
    patch_x: str,
    patch_y: str,
    timeout: float | None = None,
) -> PatchProposal:
    """Ask an editing agent to synthesize X/Y; return git-generated diff."""

    def run(worktree: Path) -> PatchProposal:
        user = f"""
You are editing a real git worktree. Do NOT output a unified diff by hand.

Task:
- Produce the conservative synthesis described by the system prompt.
- Modify ONLY `{trainer_path}`.
- If synthesis should be empty, leave files unchanged.
- After editing, final response should be only a short rationale paragraph.
Autoreason will run `git diff` itself to produce the synthesis patch.

## Patch X
```diff
{patch_x or 'NO_PATCH'}
```

## Patch Y
```diff
{patch_y or 'NO_PATCH'}
```
""".strip()
        raw = client.edit_files(SYNTHESIZER_SYSTEM_PROMPT, user, workdir=worktree, timeout=timeout)
        files = _changed_files(worktree)
        _validate_changed_files(files, {trainer_path})
        diff = _git_diff(worktree, [trainer_path]) if files else ""
        return PatchProposal(
            rationale=_extract_rationale(raw) if diff else "No file changes made by synthesis editing agent.",
            diff=diff,
            raw=raw,
        )

    return _with_git_worktree(repo_root, run)
