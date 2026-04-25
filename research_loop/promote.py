"""Promotion guardrails — pure functions over candidate eval results.

Two distinct decisions, kept separate by design (project decision 2026-04-25):

  1. `decide()` — tournament promotion (does this beat the incumbent?).
     Uses combined + recall + productness_neg_acc guardrails. `combined` itself
     stays retrieval-only (0.5 recall@1 + 0.5 mean_cosine) for comparability
     with V1 history; productness regression is a separate veto, not a blend.

  2. `is_deployable()` — shipping gate (does this checkpoint meet the bar
     to ship?). Adds absolute productness thresholds (a model with 50%
     personal-item rejection is unshippable even if retrieval is great).

Why not blend productness into combined? A weighted sum hides tradeoffs —
a +productness / -recall candidate could win the wrong way. Keeping the
metrics separate forces an explicit AND across orthogonal concerns.

Rules for `decide()` (all must pass for a non-A candidate to win):
  1. combined-metric delta over incumbent A must exceed NOISE_BAND.
  2. recall@1 must not regress beyond RECALL_REGRESSION_TOLERANCE.
  3. productness_neg_acc must not regress beyond PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE
     (when the field is present on both A and the challenger).
  4. Tie / within-noise → incumbent A wins (do-nothing bias).
  5. Missing rollback condition on a non-A candidate is an automatic veto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NOISE_BAND = 0.003                                     # combined-metric noise floor
RECALL_REGRESSION_TOLERANCE = 0.005                    # max acceptable drop in recall@1
PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE = 0.02        # 2-pt drop in personal-item rejection vetoes promotion

# Deployment-gate thresholds (separate from tournament promotion)
DEPLOY_MIN_COMBINED = 0.86
DEPLOY_MIN_PRODUCTNESS_POS_ACC = 0.97
DEPLOY_MIN_PRODUCTNESS_NEG_ACC = 0.85


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    kind: Literal["A", "B", "AB"]
    combined: float
    recall_1: float
    mean_cosine: float
    has_rollback: bool = True
    # Productness fields are optional so V1 history rows can still be
    # represented as CandidateResult; promotion only checks regression
    # when both A and the challenger expose the field.
    productness_pos_acc: float | None = None
    productness_neg_acc: float | None = None
    # Status from TrainOutcome. "timeout" candidates cannot be promoted —
    # we don't trust partial metrics for production-track decisions even
    # if they happen to look good.
    status: str = "success"


@dataclass(frozen=True)
class Decision:
    winner_id: str
    winner_kind: Literal["A", "B", "AB"]
    reason: str
    promote: bool  # True = the winner is non-A and may be applied


def is_noise_band(delta_combined: float) -> bool:
    return abs(delta_combined) <= NOISE_BAND


def regresses_recall(incumbent_recall: float, candidate_recall: float) -> bool:
    return (incumbent_recall - candidate_recall) > RECALL_REGRESSION_TOLERANCE


def regresses_productness_neg_acc(incumbent: float | None, candidate: float | None) -> bool:
    """Personal-item rejection regression check.

    Returns False (no regression) when either side lacks the metric — we
    don't penalize candidates whose runner predates the productness branch.
    """
    if incumbent is None or candidate is None:
        return False
    return (incumbent - candidate) > PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE


def decide(results: list[CandidateResult]) -> Decision:
    """Pick a winner. Incumbent A wins on tie or guardrail veto."""
    if not results:
        raise ValueError("decide() called with no results")

    incumbents = [r for r in results if r.kind == "A"]
    if len(incumbents) != 1:
        raise ValueError(f"Expected exactly one A candidate, got {len(incumbents)}")
    a = incumbents[0]

    challengers = [r for r in results if r.kind != "A"]
    best = a
    best_reason = "no challengers; incumbent A wins by default"

    for c in challengers:
        if c.status != "success":
            continue  # vetoed: timeout / failed candidates cannot be promoted
        if not c.has_rollback:
            continue  # vetoed: non-A without rollback
        delta = c.combined - a.combined
        if is_noise_band(delta):
            continue  # within noise → A wins
        if delta < 0:
            continue  # regressed
        if regresses_recall(a.recall_1, c.recall_1):
            continue  # combined up but recall@1 down
        if regresses_productness_neg_acc(a.productness_neg_acc, c.productness_neg_acc):
            continue  # combined up but personal-item rejection regressed
        if c.combined > best.combined:
            best = c
            neg_str = (
                f", neg_acc={c.productness_neg_acc:.4f} vs A={a.productness_neg_acc:.4f}"
                if c.productness_neg_acc is not None and a.productness_neg_acc is not None
                else ""
            )
            best_reason = (
                f"candidate {c.kind} beats A by Δcombined={delta:+.4f}, "
                f"recall@1={c.recall_1:.4f} vs A={a.recall_1:.4f}{neg_str}"
            )

    return Decision(
        winner_id=best.candidate_id,
        winner_kind=best.kind,
        reason=best_reason,
        promote=best.kind != "A",
    )


@dataclass(frozen=True)
class DeployVerdict:
    deployable: bool
    reasons: tuple[str, ...]  # populated with failed criteria when not deployable


def is_deployable(result: CandidateResult) -> DeployVerdict:
    """Shipping gate — does this checkpoint meet the bar to deploy?

    Independent of the tournament `decide()` — a candidate can win promotion
    over A (it's the new best) yet still not be shippable in absolute terms.
    """
    failures: list[str] = []
    if result.combined < DEPLOY_MIN_COMBINED:
        failures.append(
            f"combined={result.combined:.4f} < {DEPLOY_MIN_COMBINED}"
        )
    if result.productness_pos_acc is None or result.productness_pos_acc < DEPLOY_MIN_PRODUCTNESS_POS_ACC:
        actual = "missing" if result.productness_pos_acc is None else f"{result.productness_pos_acc:.4f}"
        failures.append(f"productness_pos_acc={actual} < {DEPLOY_MIN_PRODUCTNESS_POS_ACC}")
    if result.productness_neg_acc is None or result.productness_neg_acc < DEPLOY_MIN_PRODUCTNESS_NEG_ACC:
        actual = "missing" if result.productness_neg_acc is None else f"{result.productness_neg_acc:.4f}"
        failures.append(f"productness_neg_acc={actual} < {DEPLOY_MIN_PRODUCTNESS_NEG_ACC}")
    return DeployVerdict(deployable=not failures, reasons=tuple(failures))
