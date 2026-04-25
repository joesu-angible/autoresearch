"""Rule-based judges for the autoreason tournament.

Three rubric scorers, each producing a float in [0, 1]:
  - clarity:     hypothesis is concrete, expected_metric is quantifiable
  - risk:        risks are enumerated, rollback condition exists
  - prior:       expected_metric is consistent with historical results_v2.tsv

Final ranking aggregates the three with equal weight; ties resolve toward
the do-nothing incumbent A (judges cannot push a tie toward action).

LLM-based judges are out of scope here; the `Judge` Protocol below is the
plug-in seam for a future LLM-backed implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from research_loop.candidate import Candidate

MIN_HYPOTHESIS_LEN = 20
MIN_EXPECTED_METRIC_LEN = 3


class Judge(Protocol):
    """Plug-in seam — any object with these methods can replace rule-based scorers."""

    def score(self, candidate: Candidate) -> float: ...


def score_clarity(c: Candidate) -> float:
    """Clarity = hypothesis length + presence of a numeric/quantifiable expected metric."""
    hyp = c.hypothesis.strip()
    em = c.expected_metric.strip()
    score = 0.0
    if len(hyp) >= MIN_HYPOTHESIS_LEN:
        score += 0.5
    if any(tok in em for tok in ("+", "-", "%", "0.")) and len(em) >= MIN_EXPECTED_METRIC_LEN:
        score += 0.5
    return score


def score_risk(c: Candidate) -> float:
    """Risk = explicit risks listed AND rollback condition present.

    Incumbent A (do-nothing) gets full marks — there is nothing to roll back.
    """
    if c.kind == "A":
        return 1.0
    risk_score = 0.5 if c.risks else 0.0
    rollback_score = 0.5 if c.rollback.strip() and c.rollback.strip().lower() != "n/a" else 0.0
    return risk_score + rollback_score


def score_prior_evidence(c: Candidate, history_results: list[dict] | None = None) -> float:
    """Consistency with historical results_v2.tsv rows.

    Today: returns 1.0 if any evidence_refs are listed (proves the proposer
    actually consulted history); 0.5 if not. A future LLM judge can fill in
    the semantic check.
    """
    if c.kind == "A":
        return 1.0
    if c.evidence_refs:
        return 1.0
    return 0.5


def aggregate(c: Candidate, history_results: list[dict] | None = None) -> float:
    return (
        score_clarity(c)
        + score_risk(c)
        + score_prior_evidence(c, history_results)
    ) / 3.0


@dataclass(frozen=True)
class JudgeOrdering:
    candidate_id: str
    kind: str
    score: float


def rank(
    candidates: Iterable[Candidate],
    history_results: list[dict] | None = None,
) -> list[JudgeOrdering]:
    """Sort candidates highest score first; ties resolve to incumbent A."""
    scored = [
        JudgeOrdering(c.id, c.kind, aggregate(c, history_results))
        for c in candidates
    ]
    # Tiebreaker key: kind=='A' first when scores equal
    return sorted(scored, key=lambda o: (-o.score, 0 if o.kind == "A" else 1))
