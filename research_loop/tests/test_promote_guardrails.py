"""T3 verification: promotion guardrails + deployment gate."""

from __future__ import annotations

import pytest

from research_loop.promote import (
    CandidateResult,
    DEPLOY_MIN_COMBINED,
    DEPLOY_MIN_PRODUCTNESS_NEG_ACC,
    DEPLOY_MIN_PRODUCTNESS_POS_ACC,
    NOISE_BAND,
    PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE,
    RECALL_REGRESSION_TOLERANCE,
    decide,
    is_deployable,
    is_noise_band,
    regresses_productness_neg_acc,
    regresses_recall,
)


def _a(combined=0.86, recall_1=0.90, mean_cosine=0.81):
    return CandidateResult(
        candidate_id="a-id",
        kind="A",
        combined=combined,
        recall_1=recall_1,
        mean_cosine=mean_cosine,
    )


def _b(combined=0.87, recall_1=0.91, mean_cosine=0.82, has_rollback=True, cid="b-id"):
    return CandidateResult(
        candidate_id=cid,
        kind="B",
        combined=combined,
        recall_1=recall_1,
        mean_cosine=mean_cosine,
        has_rollback=has_rollback,
    )


def test_is_noise_band():
    assert is_noise_band(0.0)
    assert is_noise_band(NOISE_BAND)
    assert is_noise_band(-NOISE_BAND)
    assert not is_noise_band(NOISE_BAND + 1e-6)


def test_regresses_recall():
    eps = 1e-9
    assert not regresses_recall(0.90, 0.90)
    # just inside tolerance (drop slightly less than tolerance) → not a regression
    assert not regresses_recall(0.90, 0.90 - RECALL_REGRESSION_TOLERANCE + eps)
    # clearly past tolerance → regression
    assert regresses_recall(0.90, 0.90 - RECALL_REGRESSION_TOLERANCE - 1e-4)


def test_a_wins_when_no_challengers():
    decision = decide([_a()])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_a_wins_on_noise_band_tie():
    a = _a(combined=0.860)
    # strictly inside the noise band → tie → A wins
    b = _b(combined=0.860 + NOISE_BAND - 1e-6)
    decision = decide([a, b])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_b_wins_on_clear_improvement():
    a = _a(combined=0.860, recall_1=0.900)
    b = _b(combined=0.870, recall_1=0.905)
    decision = decide([a, b])
    assert decision.winner_kind == "B"
    assert decision.promote is True


def test_recall_regression_vetoes_combined_win():
    a = _a(combined=0.860, recall_1=0.910)
    # combined improves clearly but recall@1 drops > tolerance
    b = _b(
        combined=0.870,
        recall_1=0.910 - RECALL_REGRESSION_TOLERANCE - 1e-3,
        mean_cosine=0.85,
    )
    decision = decide([a, b])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_missing_rollback_vetoes_b():
    a = _a()
    b = _b(combined=0.90, has_rollback=False)
    decision = decide([a, b])
    assert decision.winner_kind == "A"


def test_best_among_multiple_challengers_wins():
    a = _a(combined=0.860, recall_1=0.900)
    b = _b(combined=0.870, recall_1=0.905, cid="b-id")
    ab = CandidateResult(
        candidate_id="ab-id",
        kind="AB",
        combined=0.880,
        recall_1=0.906,
        mean_cosine=0.83,
    )
    decision = decide([a, b, ab])
    assert decision.winner_id == "ab-id"
    assert decision.winner_kind == "AB"


def test_decide_requires_exactly_one_a():
    with pytest.raises(ValueError):
        decide([_b()])
    with pytest.raises(ValueError):
        decide([_a(), _a()])


def test_decide_empty_raises():
    with pytest.raises(ValueError):
        decide([])


# ---------------------------------------------------------------------------
# Productness regression guardrail (Option A — separate veto, not blended)
# ---------------------------------------------------------------------------

def _b_with_prod(combined=0.87, recall_1=0.91, neg_acc=0.85, pos_acc=0.98, cid="b-id"):
    return CandidateResult(
        candidate_id=cid, kind="B",
        combined=combined, recall_1=recall_1, mean_cosine=0.82,
        productness_pos_acc=pos_acc, productness_neg_acc=neg_acc,
    )


def _a_with_prod(combined=0.86, recall_1=0.90, neg_acc=0.85, pos_acc=0.98):
    return CandidateResult(
        candidate_id="a-id", kind="A",
        combined=combined, recall_1=recall_1, mean_cosine=0.81,
        productness_pos_acc=pos_acc, productness_neg_acc=neg_acc,
    )


def test_regresses_productness_neg_acc_returns_false_when_either_missing():
    """Old runs without productness fields must not be vetoed."""
    assert not regresses_productness_neg_acc(None, 0.5)
    assert not regresses_productness_neg_acc(0.85, None)
    assert not regresses_productness_neg_acc(None, None)


def test_regresses_productness_neg_acc_threshold():
    eps = 1e-9
    # exactly tolerance → not a regression
    assert not regresses_productness_neg_acc(0.85, 0.85 - PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE + eps)
    # past tolerance → regression
    assert regresses_productness_neg_acc(0.85, 0.85 - PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE - 1e-3)


def test_productness_neg_acc_regression_vetoes_b():
    """B improves combined and recall, but tanks personal-item rejection → A wins."""
    a = _a_with_prod(combined=0.860, recall_1=0.900, neg_acc=0.85)
    b = _b_with_prod(combined=0.880, recall_1=0.905, neg_acc=0.82)  # -3 pts on neg_acc
    decision = decide([a, b])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_productness_neg_acc_within_tolerance_does_not_veto():
    a = _a_with_prod(combined=0.860, recall_1=0.900, neg_acc=0.85)
    b = _b_with_prod(combined=0.880, recall_1=0.905, neg_acc=0.84)  # -1 pt, within 2pt tolerance
    decision = decide([a, b])
    assert decision.winner_kind == "B"
    assert decision.promote is True


def test_productness_neg_acc_improvement_does_not_block():
    a = _a_with_prod(combined=0.860, recall_1=0.900, neg_acc=0.85)
    b = _b_with_prod(combined=0.880, recall_1=0.905, neg_acc=0.92)  # better
    decision = decide([a, b])
    assert decision.winner_kind == "B"


def test_decide_works_when_productness_fields_absent_on_both_sides():
    """V1-history compatibility: no productness on either side → decide is unaffected."""
    a = CandidateResult("a", "A", combined=0.86, recall_1=0.90, mean_cosine=0.81)
    b = CandidateResult("b", "B", combined=0.88, recall_1=0.905, mean_cosine=0.83)
    decision = decide([a, b])
    assert decision.winner_kind == "B"


# ---------------------------------------------------------------------------
# Deployment gate (separate from tournament promotion)
# ---------------------------------------------------------------------------

def test_is_deployable_full_pass():
    r = _b_with_prod(combined=0.88, neg_acc=0.90, pos_acc=0.98, cid="r")
    verdict = is_deployable(r)
    assert verdict.deployable is True
    assert verdict.reasons == ()


def test_is_deployable_combined_too_low():
    r = _b_with_prod(combined=DEPLOY_MIN_COMBINED - 0.01, neg_acc=0.90, pos_acc=0.99)
    verdict = is_deployable(r)
    assert verdict.deployable is False
    assert any("combined" in reason for reason in verdict.reasons)


def test_is_deployable_neg_acc_too_low():
    r = _b_with_prod(combined=0.88, neg_acc=DEPLOY_MIN_PRODUCTNESS_NEG_ACC - 0.01, pos_acc=0.99)
    verdict = is_deployable(r)
    assert verdict.deployable is False
    assert any("neg_acc" in reason for reason in verdict.reasons)


def test_is_deployable_pos_acc_too_low():
    r = _b_with_prod(combined=0.88, pos_acc=DEPLOY_MIN_PRODUCTNESS_POS_ACC - 0.01, neg_acc=0.90)
    verdict = is_deployable(r)
    assert verdict.deployable is False
    assert any("pos_acc" in reason for reason in verdict.reasons)


def test_is_deployable_missing_productness_fails():
    """A V1-style result without productness cannot be deployed under V2 rules."""
    r = CandidateResult("v1", "A", combined=0.88, recall_1=0.92, mean_cosine=0.84)
    verdict = is_deployable(r)
    assert verdict.deployable is False
    assert any("pos_acc" in reason for reason in verdict.reasons)
    assert any("neg_acc" in reason for reason in verdict.reasons)


def test_is_deployable_collects_all_failures():
    """Multiple failed criteria all surface in verdict.reasons."""
    r = _b_with_prod(combined=0.10, neg_acc=0.10, pos_acc=0.10)
    verdict = is_deployable(r)
    assert verdict.deployable is False
    assert len(verdict.reasons) == 3


# ---------------------------------------------------------------------------
# T3 — N>2 (sweep mode, issue #9 Goal 3): decide() over 1 A + many Bs
# ---------------------------------------------------------------------------

def test_decide_n_variants_picks_highest_combined():
    """Sweep with 5 B challengers: best-combined wins, others ignored."""
    a = _a(combined=0.860, recall_1=0.900)
    bs = [
        _b(combined=0.865, recall_1=0.905, cid="b0"),
        _b(combined=0.872, recall_1=0.906, cid="b1"),
        _b(combined=0.880, recall_1=0.910, cid="b2"),  # winner
        _b(combined=0.870, recall_1=0.905, cid="b3"),
        _b(combined=0.866, recall_1=0.905, cid="b4"),
    ]
    decision = decide([a, *bs])
    assert decision.winner_id == "b2"
    assert decision.promote is True


def test_decide_n_variants_all_regress_a_wins():
    """All N challengers worse than A → A wins by default."""
    a = _a(combined=0.880, recall_1=0.910)
    bs = [_b(combined=0.860 + 0.001 * i, recall_1=0.900, cid=f"b{i}") for i in range(5)]
    decision = decide([a, *bs])
    assert decision.winner_kind == "A"
    assert decision.promote is False


def test_decide_n_variants_skips_guardrail_violator_picks_runner_up():
    """Highest-combined B regresses recall@1 → next-highest wins."""
    a = _a(combined=0.860, recall_1=0.910)
    bs = [
        _b(combined=0.866, recall_1=0.908, cid="ok-low"),
        _b(combined=0.872, recall_1=0.908, cid="ok-mid"),
        _b(combined=0.880, recall_1=0.880, cid="vetoed"),  # +combined but recall regression > 0.005
        _b(combined=0.875, recall_1=0.909, cid="ok-runner-up"),  # should win
    ]
    decision = decide([a, *bs])
    assert decision.winner_id == "ok-runner-up"


def test_decide_n_variants_within_noise_band_skipped():
    """Variants within noise of A but with higher recall don't auto-promote."""
    a = _a(combined=0.870, recall_1=0.900)
    # All 4 strictly inside the noise band (NOISE_BAND/2 avoids float-precision
    # at the boundary — see existing test at line 69).
    bs = [
        _b(combined=0.870 + NOISE_BAND / 2, recall_1=0.910, cid=f"b{i}")
        for i in range(4)
    ]
    decision = decide([a, *bs])
    assert decision.winner_kind == "A"


def test_decide_n_variants_failed_status_excluded():
    """Variants with status != 'success' are excluded; runner-up succeeds."""
    a = _a(combined=0.860, recall_1=0.900)
    high_but_failed = CandidateResult(
        candidate_id="failed-best", kind="B",
        combined=0.890, recall_1=0.915, mean_cosine=0.83,
        status="timeout",
    )
    succeeding = _b(combined=0.875, recall_1=0.910, cid="ok-winner")
    decision = decide([a, high_but_failed, succeeding])
    assert decision.winner_id == "ok-winner"
