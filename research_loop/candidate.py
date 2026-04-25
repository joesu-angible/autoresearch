"""Candidate / Outcome / Decision schema for the autoreason tournament.

A round produces three candidates:
  - A = do-nothing incumbent (current best of results_v2.tsv)
  - B = patched candidate (the new experiment)
  - AB = conservative synthesis of A and B

After running each candidate, an Outcome record stores the objective metrics.
After the round closes, a Decision record stores the promote/reject verdict.

All three record types share `research_loop/history.jsonl`, discriminated by
the `record_type` field. Downstream readers filter on record_type.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

CandidateKind = Literal["A", "B", "AB"]
TargetName = Literal["student_v2", "dino_v2"]
RecordType = Literal["candidate", "outcome", "decision", "critique", "patch_proposal", "synthesis", "outcome_started"]

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


def new_round_id() -> str:
    """Round id = unix-time-prefixed uuid; sortable + globally unique."""
    return f"r{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    round_id: str = ""
    record_type: RecordType = "candidate"

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
        # tolerate older records without round_id / record_type
        data.setdefault("round_id", "")
        data.setdefault("record_type", "candidate")
        return cls(**data)


@dataclass
class OutcomeStartedRecord:
    """Marks the moment a candidate began training.

    Written by cmd_autoreason immediately before adapter.train() runs. Paired
    with an Outcome record (same candidate_id) on completion. A candidate
    with an OutcomeStartedRecord but no matching Outcome is "unfinished" —
    the runner crashed mid-training. Resume uses this state machine to know
    which candidates need to be re-run vs which are already done.
    """

    candidate_id: str
    round_id: str
    target: TargetName
    pass_index: int      # autoreason pass number; 0 for non-autoreason runs
    kind: CandidateKind
    run_id: str = ""     # autoreason run_id; "" for non-autoreason runs
    record_type: RecordType = "outcome_started"
    started_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "OutcomeStartedRecord":
        return cls(**json.loads(line))


@dataclass
class Outcome:
    """Objective result of running a Candidate end-to-end."""

    candidate_id: str
    round_id: str
    target: TargetName
    status: Literal["success", "failed", "noop"]
    metrics: dict[str, float]      # combined, recall_1, recall_5, mean_cosine, productness_*
    elapsed_seconds: float
    log_path: str
    metrics_json_path: str
    record_type: RecordType = "outcome"
    completed_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "Outcome":
        data = json.loads(line)
        return cls(**data)


@dataclass
class Decision:
    """Round-level promote/reject verdict from research_loop.promote.decide."""

    round_id: str
    target: TargetName
    winner_id: str
    winner_kind: CandidateKind
    promote: bool
    reason: str
    deployable: bool
    deploy_failures: tuple[str, ...] = ()
    record_type: RecordType = "decision"
    decided_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        d = asdict(self)
        # tuple → list for json serialization
        d["deploy_failures"] = list(d["deploy_failures"])
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "Decision":
        data = json.loads(line)
        data["deploy_failures"] = tuple(data.get("deploy_failures", ()))
        return cls(**data)


@dataclass
class CritiqueRecord:
    """Audit record of one Critic LLM call."""

    round_id: str
    target: TargetName
    pass_index: int
    summary: str
    problems: list[str]
    raw: str  # full LLM response text
    record_type: RecordType = "critique"
    created_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "CritiqueRecord":
        return cls(**json.loads(line))


@dataclass
class PatchProposalRecord:
    """Audit record of one Author B LLM call."""

    round_id: str
    target: TargetName
    pass_index: int
    candidate_id: str  # the B candidate this patch is attached to
    rationale: str
    diff: str
    raw: str
    record_type: RecordType = "patch_proposal"
    created_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "PatchProposalRecord":
        return cls(**json.loads(line))


@dataclass
class SynthesisRecord:
    """Audit record of one Synthesizer LLM call."""

    round_id: str
    target: TargetName
    pass_index: int
    candidate_id: str  # the AB candidate this synthesis is attached to
    rationale: str
    diff: str
    raw: str
    record_type: RecordType = "synthesis"
    created_at: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> "SynthesisRecord":
        return cls(**json.loads(line))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

HistoryRecord = "Candidate | Outcome | Decision | CritiqueRecord | PatchProposalRecord | SynthesisRecord | OutcomeStartedRecord"


def append_history(history_path: Path, record: HistoryRecord) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as f:
        f.write(record.to_jsonl() + "\n")


def _iter_records(history_path: Path) -> Iterator[dict]:
    if not history_path.exists():
        return
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_history(history_path: Path) -> Iterator[Candidate]:
    """Yield only Candidate records (back-compat with earlier callers)."""
    for raw in _iter_records(history_path):
        if raw.get("record_type", "candidate") == "candidate":
            raw.setdefault("round_id", "")
            raw.setdefault("record_type", "candidate")
            yield Candidate(**raw)


def read_outcomes(history_path: Path, round_id: str | None = None) -> Iterator[Outcome]:
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "outcome":
            continue
        if round_id is not None and raw.get("round_id") != round_id:
            continue
        yield Outcome(**raw)


def read_decisions(history_path: Path, round_id: str | None = None) -> Iterator[Decision]:
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "decision":
            continue
        if round_id is not None and raw.get("round_id") != round_id:
            continue
        raw["deploy_failures"] = tuple(raw.get("deploy_failures", ()))
        yield Decision(**raw)


def read_critiques(history_path: Path, round_id: str | None = None) -> Iterator[CritiqueRecord]:
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "critique":
            continue
        if round_id is not None and raw.get("round_id") != round_id:
            continue
        yield CritiqueRecord(**raw)


def read_patch_proposals(history_path: Path, round_id: str | None = None) -> Iterator[PatchProposalRecord]:
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "patch_proposal":
            continue
        if round_id is not None and raw.get("round_id") != round_id:
            continue
        yield PatchProposalRecord(**raw)


def read_syntheses(history_path: Path, round_id: str | None = None) -> Iterator[SynthesisRecord]:
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "synthesis":
            continue
        if round_id is not None and raw.get("round_id") != round_id:
            continue
        yield SynthesisRecord(**raw)


def read_outcomes_started(history_path: Path, run_id: str | None = None) -> Iterator[OutcomeStartedRecord]:
    """Yield OutcomeStartedRecord entries; optionally filter by run_id."""
    for raw in _iter_records(history_path):
        if raw.get("record_type") != "outcome_started":
            continue
        if run_id is not None and raw.get("run_id", "") != run_id:
            continue
        yield OutcomeStartedRecord(**raw)


def find_unfinished_candidates(history_path: Path, run_id: str) -> list[Candidate]:
    """Return Candidates that started but never completed for this run_id.

    Pairing rule: the *latest* outcome_started for a candidate_id is matched
    against any outcome with the same candidate_id. If outcome is missing,
    the candidate is unfinished. Tolerates duplicate outcome_started records
    (e.g. resume → recrash → resume).

    Backward compatible: a history.jsonl with no outcome_started records
    returns []; nothing was tracked, so nothing is reported unfinished.
    """
    started_ids: set[str] = set()
    for s in read_outcomes_started(history_path, run_id=run_id):
        started_ids.add(s.candidate_id)
    if not started_ids:
        return []
    completed_ids = {o.candidate_id for o in read_outcomes(history_path)}
    unfinished_ids = started_ids - completed_ids
    if not unfinished_ids:
        return []
    by_id = {c.id: c for c in read_history(history_path) if c.id in unfinished_ids}
    # Preserve start order for deterministic re-run sequencing
    ordered: list[Candidate] = []
    seen: set[str] = set()
    for s in read_outcomes_started(history_path, run_id=run_id):
        if s.candidate_id in unfinished_ids and s.candidate_id not in seen:
            seen.add(s.candidate_id)
            if s.candidate_id in by_id:
                ordered.append(by_id[s.candidate_id])
    return ordered


def find_candidate(history_path: Path, candidate_id: str) -> Candidate | None:
    for c in read_history(history_path):
        if c.id == candidate_id:
            return c
    return None
