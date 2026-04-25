"""Candidate schema for the autoreason tournament.

A round produces three candidates:
  - A = do-nothing incumbent (current best of results_v2.tsv)
  - B = patched candidate (the new experiment)
  - AB = conservative synthesis of A and B

Persisted as JSONL in research_loop/history.jsonl (one record per candidate,
plus a separate "round" record summarising the tournament outcome).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

CandidateKind = Literal["A", "B", "AB"]
TargetName = Literal["student_v2", "dino_v2"]

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "kind",
    "target",
    "hypothesis",
    "expected_metric",
    "changed_files",
    "risks",
    "rollback",
)


@dataclass
class Candidate:
    kind: CandidateKind
    target: TargetName
    hypothesis: str
    expected_metric: str
    changed_files: list[str]
    risks: list[str]
    rollback: str
    parent_incumbent_id: str | None = None
    patch: str = ""  # unified diff; empty for kind="A" (do-nothing)
    evidence_refs: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if self.kind == "A" and self.patch.strip():
            raise ValueError("Candidate kind='A' (do-nothing) must have an empty patch.")
        if self.kind in ("B", "AB") and not self.patch.strip():
            raise ValueError(f"Candidate kind={self.kind!r} requires a non-empty patch.")
        if not self.rollback.strip():
            raise ValueError("rollback condition is required.")
        if not self.changed_files and self.kind != "A":
            raise ValueError("changed_files must be non-empty for B/AB candidates.")

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "Candidate":
        data = json.loads(line)
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(**data)


def append_history(history_path: Path, candidate: Candidate) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as f:
        f.write(candidate.to_jsonl() + "\n")


def read_history(history_path: Path) -> Iterator[Candidate]:
    if not history_path.exists():
        return
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Candidate.from_jsonl(line)
