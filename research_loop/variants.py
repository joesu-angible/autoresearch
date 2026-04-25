"""Variants file loader for `propose --variants <path>` (issue #9 Goal 3).

A variants file is JSONL — one JSON object per line — describing N candidate
patches that should run as a single tournament round against the current
incumbent A. Each line specifies the per-candidate fields that `cmd_propose`
cannot fill in for the operator: hypothesis, expected_metric, changed_files,
risks, rollback, and the unified-diff patch itself. The loader validates the
schema; patch validity (`git apply --check`) is deferred to `cmd_run`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class VariantSpec(TypedDict):
    hypothesis: str
    expected_metric: str
    changed_files: list[str]
    risks: list[str]
    rollback: str
    patch: str


REQUIRED_FIELDS: tuple[str, ...] = (
    "hypothesis",
    "expected_metric",
    "changed_files",
    "risks",
    "rollback",
    "patch",
)


def load_variants(path: Path) -> list[VariantSpec]:
    """Parse a JSONL variants file. Skips blank lines.

    Raises ValueError with the offending 1-based line number on any malformed
    JSON or missing required field. Returns an empty list for a file containing
    only blank lines — the caller decides whether that's an error.
    """
    specs: list[VariantSpec] = []
    text = path.read_text()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"variants file {path}: line {lineno}: malformed JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"variants file {path}: line {lineno}: expected JSON object, got {type(obj).__name__}")
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            raise ValueError(f"variants file {path}: line {lineno}: missing required fields {missing}")
        specs.append(VariantSpec(
            hypothesis=obj["hypothesis"],
            expected_metric=obj["expected_metric"],
            changed_files=list(obj["changed_files"]),
            risks=list(obj["risks"]),
            rollback=obj["rollback"],
            patch=obj["patch"],
        ))
    return specs
