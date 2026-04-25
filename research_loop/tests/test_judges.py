"""T10 verification: rule-based judges + ranking tie-break to A."""

from __future__ import annotations

from research_loop.candidate import Candidate
from research_loop.judges import (
    aggregate,
    rank,
    score_clarity,
    score_prior_evidence,
    score_risk,
)


def _a():
    return Candidate(
        kind="A",
        target="student_v2",
        # quantifiable expected_metric ("0.") so clarity scores 1.0; this lets
        # us construct a tie scenario against a strong B candidate
        hypothesis="do nothing — keep current incumbent baseline",
        expected_metric="combined Δ +0.000 (no change)",
        changed_files=[],
        risks=[],
        rollback="N/A",
        patch="",
    )


def _b(**over):
    base = dict(
        kind="B",
        target="student_v2",
        hypothesis="raise commodity_ratio from 0.15 to 0.20 to widen distill coverage",
        expected_metric="combined +0.004",
        changed_files=["student_finetune/train_v2.py"],
        risks=["may dilute label signal"],
        rollback="combined < 0.855",
        patch="--- a\n+++ b\n",
        evidence_refs=["results_v2.tsv:row=8"],
    )
    base.update(over)
    return Candidate(**base)


def test_clarity_rewards_quantifiable_metric():
    assert score_clarity(_b()) == 1.0
    weak = _b(hypothesis="more data", expected_metric="better")
    assert score_clarity(weak) == 0.0


def test_risk_full_marks_for_a():
    assert score_risk(_a()) == 1.0


def test_risk_requires_both_risks_and_rollback_for_b():
    no_risks = _b(risks=[])
    assert score_risk(no_risks) == 0.5
    no_rollback = _b(rollback="N/A")  # explicit N/A is treated as missing for non-A
    assert score_risk(no_rollback) == 0.5


def test_prior_evidence_rewards_history_refs():
    assert score_prior_evidence(_b()) == 1.0
    no_refs = _b(evidence_refs=[])
    assert score_prior_evidence(no_refs) == 0.5


def test_rank_orders_by_aggregate():
    a = _a()
    b_strong = _b()
    b_weak = _b(hypothesis="more data", expected_metric="better", risks=[], evidence_refs=[])
    ordering = rank([b_weak, a, b_strong])
    # A is full marks; b_strong full marks too; tie → A first
    assert ordering[0].kind == "A"
    assert ordering[1].kind == "B"
    assert ordering[1].candidate_id == b_strong.id
    # b_weak last
    assert ordering[-1].candidate_id == b_weak.id


def test_aggregate_in_unit_interval():
    for c in (_a(), _b()):
        s = aggregate(c)
        assert 0.0 <= s <= 1.0


# T3 — N>2 ranking (issue #9 Goal 3): sweep round with many B variants
def test_rank_orders_n_variants_with_a_tiebreak():
    """Sweep mode: 1 A + 5 strong B variants. All score full marks → A first
    by tie-break, all five Bs after."""
    a = _a()
    bs = [
        _b(hypothesis=f"variant {i} with quantifiable target",
           expected_metric=f"combined +0.00{i+1}")
        for i in range(5)
    ]
    ordering = rank([*bs, a])
    assert len(ordering) == 6
    assert ordering[0].kind == "A"  # tie → A first
    assert all(o.kind == "B" for o in ordering[1:])
    # Strict descending by score (with A first on tie)
    scores = [o.score for o in ordering]
    assert scores == sorted(scores, reverse=True)


def test_rank_n_variants_with_varying_quality():
    """Mix of strong and weak Bs in N>2 sweep — weakest end up last."""
    a = _a()
    strong = _b(hypothesis="strong B with full-marks rationale")
    medium = _b(hypothesis="medium B", evidence_refs=[])  # no prior evidence
    weak = _b(hypothesis="weak", expected_metric="better", risks=[], evidence_refs=[])
    ordering = rank([weak, a, medium, strong])
    assert len(ordering) == 4
    assert ordering[0].kind == "A"
    # strong B should rank above medium and weak among the Bs
    b_ordering = [o for o in ordering if o.kind == "B"]
    assert b_ordering[0].candidate_id == strong.id
    assert b_ordering[-1].candidate_id == weak.id
