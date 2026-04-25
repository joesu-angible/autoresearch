"""Apply / revert unified diffs via `git apply` for autoreason rounds.

Every autoreason pass mutates the working tree (Author B's patch, then AB's
synthesis), runs training, then must restore the tree exactly. Doing this
through `git apply` + `git apply -R` instead of file-level edits gives us:

  - Atomic apply: `git apply --check` runs first; if any hunk fails, nothing
    is applied. No partial mutations to clean up.
  - Reversible: `git apply -R` is the official inverse of `git apply` for
    the same diff. Same patch in / same patch out.
  - V2 safety: we parse the diff's file paths first and reject any that
    match V1_FORBIDDEN_PATHS. The LLM Author B's system prompt also forbids
    V1 edits, but defense in depth — never trust the LLM's word.
  - No working-tree pollution between rounds: try/finally guarantees revert
    even on training-side exceptions.

Usage:
    with apply_patch(diff_text, repo=Path("/path/to/repo")):
        # working tree mutated; run training, eval, etc.
        ...
    # working tree restored, regardless of how the inner block exited.
"""

from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from research_loop.targets._base import V1_FORBIDDEN_PATHS

# Match unified-diff file headers: "+++ b/path/to/file" or "--- a/path/to/file"
# and the "diff --git" line to be tolerant of `git diff` output.
_DIFF_PATH_RE = re.compile(
    r"^(?:\+\+\+ b/|--- a/|diff --git a/)(?P<path>\S+)",
    re.MULTILINE,
)


def diff_touches_v1(diff_text: str) -> list[str]:
    """Return the V1_FORBIDDEN_PATHS that this diff would modify (empty list = safe).

    Parses the diff's file headers; any path matching V1_FORBIDDEN_PATHS
    (suffix or exact) is reported. Designed to be called *before* `git apply`.
    """
    paths = {m.group("path") for m in _DIFF_PATH_RE.finditer(diff_text)}
    # Strip the "b/" prefix on diff --git lines that captured the second path
    paths = {p.removeprefix("b/") for p in paths}
    forbidden_hits: list[str] = []
    for p in paths:
        for forbidden in V1_FORBIDDEN_PATHS:
            if p == forbidden or p.endswith("/" + forbidden) or p.endswith(forbidden):
                forbidden_hits.append(p)
                break
    return forbidden_hits


def _git(args: list[str], *, cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def _check_clean_tree(repo: Path) -> None:
    """Refuse to apply if the working tree has uncommitted changes —
    we'd lose the user's work on revert."""
    res = _git(["status", "--porcelain"], cwd=repo)
    if res.returncode != 0:
        raise RuntimeError(f"git status failed in {repo}: {res.stderr}")
    if res.stdout.strip():
        raise RuntimeError(
            "Working tree is dirty; commit or stash before applying patches.\n"
            f"git status:\n{res.stdout}"
        )


@contextmanager
def apply_patch(diff_text: str, *, repo: Path, require_clean_tree: bool = True) -> Iterator[None]:
    """Apply a unified diff; revert on context exit.

    Raises ValueError if the diff touches V1_FORBIDDEN_PATHS (no mutation).
    Raises subprocess.CalledProcessError if git apply --check rejects the
    diff (malformed, conflicts, etc.) — also no mutation.
    """
    if not diff_text.strip():
        # Empty diff = no-op (matches kind="A" do-nothing semantics)
        yield
        return

    forbidden = diff_touches_v1(diff_text)
    if forbidden:
        raise ValueError(
            f"Patch touches V1_FORBIDDEN_PATHS files: {forbidden}. "
            f"Tournament patches must only modify V2 files."
        )

    if require_clean_tree:
        _check_clean_tree(repo)

    # Validate before mutating — `git apply --check` exits non-zero on conflict.
    check = _git(["apply", "--check"], cwd=repo, stdin=diff_text)
    if check.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=check.returncode,
            cmd=["git", "apply", "--check"],
            output=check.stdout,
            stderr=check.stderr,
        )

    apply_res = _git(["apply"], cwd=repo, stdin=diff_text)
    if apply_res.returncode != 0:  # pragma: no cover — should be caught by --check
        raise subprocess.CalledProcessError(
            returncode=apply_res.returncode,
            cmd=["git", "apply"],
            output=apply_res.stdout,
            stderr=apply_res.stderr,
        )

    try:
        yield
    finally:
        revert = _git(["apply", "-R"], cwd=repo, stdin=diff_text)
        if revert.returncode != 0:  # pragma: no cover — surfaces the bug, doesn't swallow
            raise RuntimeError(
                f"Patch revert failed; working tree may be dirty.\n"
                f"stderr: {revert.stderr}\n"
                f"Original patch:\n{diff_text}"
            )
