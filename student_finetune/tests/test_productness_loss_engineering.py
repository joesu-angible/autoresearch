"""Verify label smoothing + focal weighting on the productness BCE.

Tests the inline loss block in `train.py::run_train_epoch`. Rather than
re-running the whole epoch loop (covered by the integration smoke), we
re-implement the same math here and check invariants:

  - smoothing maps hard 0/1 → soft targets correctly
  - focal weight down-weights confident-correct samples
  - gamma=0 + smoothing=0 collapses to plain BCE
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
import torch.nn.functional as F


def loss_block(
    logits: "torch.Tensor",
    y_hard: "torch.Tensor",
    eps_pos: float,
    eps_neg: float,
    gamma: float,
) -> "torch.Tensor":
    """Mirror of the productness loss block in train.py::run_train_epoch."""
    if eps_pos > 0.0 or eps_neg > 0.0:
        y_target = y_hard * (1.0 - eps_pos) + (1.0 - y_hard) * eps_neg
    else:
        y_target = y_hard
    per_sample = F.binary_cross_entropy_with_logits(logits, y_target, reduction="none")
    if gamma > 0.0:
        with torch.no_grad():
            p = torch.sigmoid(logits)
            p_t = y_hard * p + (1.0 - y_hard) * (1.0 - p)
            focal_w = (1.0 - p_t).pow(gamma)
        return (per_sample * focal_w).mean()
    return per_sample.mean()


def test_plain_bce_when_all_zero():
    """gamma=0, eps=0 → identical to F.binary_cross_entropy_with_logits."""
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    ours = loss_block(logits, y, 0.0, 0.0, 0.0)
    ref = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(ours, ref, atol=1e-7)


def test_label_smoothing_changes_target_correctly():
    """Asymmetric smoothing: y=1 → 0.95, y=0 → 0.02 with eps_pos=0.05, eps_neg=0.02."""
    y_hard = torch.tensor([1.0, 0.0, 1.0, 0.0])
    eps_pos, eps_neg = 0.05, 0.02
    smoothed = y_hard * (1.0 - eps_pos) + (1.0 - y_hard) * eps_neg
    expected = torch.tensor([0.95, 0.02, 0.95, 0.02])
    assert torch.allclose(smoothed, expected, atol=1e-7)


def test_label_smoothing_caps_confident_loss_floor():
    """The real property of label smoothing: the BCE no longer drives toward
    zero loss as logits → ∞ on the correct class. Confident-correct samples
    pay a non-zero floor that grows with the smoothing epsilon."""
    # Very confident correct positive: logit=10, target=1
    logit_correct = torch.tensor([10.0])
    y_pos = torch.tensor([1.0])
    plain = loss_block(logit_correct, y_pos, 0.0, 0.0, 0.0)
    smoothed = loss_block(logit_correct, y_pos, eps_pos=0.05, eps_neg=0.0, gamma=0.0)
    # Plain BCE on a confident-correct sample is near zero
    assert plain.item() < 0.01
    # Smoothed BCE on the same sample is non-trivial — this is the regularization
    # signal that prevents the head from being arbitrarily overconfident.
    assert smoothed.item() > 0.05


def test_focal_downweights_easy_examples():
    """High-confidence-correct samples should contribute almost nothing under γ=2."""
    # Sample 0: very confident correct positive (logit=10, y=1) — easy
    # Sample 1: confident wrong negative   (logit=10, y=0) — hard
    logits = torch.tensor([10.0, 10.0])
    y_hard = torch.tensor([1.0, 0.0])
    plain = loss_block(logits, y_hard, 0.0, 0.0, gamma=0.0)
    focal = loss_block(logits, y_hard, 0.0, 0.0, gamma=2.0)
    # Focal should *not* be larger than plain (the easy sample is suppressed,
    # the hard sample's gradient is preserved or amplified relative to it).
    assert focal < plain
    # Concretely: easy sample's focal weight is (1 - sigmoid(10))^2 ≈ 2e-9, near zero
    p_easy = torch.sigmoid(torch.tensor(10.0))
    focal_w_easy = (1.0 - p_easy) ** 2
    assert focal_w_easy < 1e-7


def test_focal_preserves_hard_examples():
    """When all examples are equally hard (logits near 0), focal weight is moderate."""
    logits = torch.zeros(8)  # p=0.5 everywhere → p_t=0.5 → focal_w=0.25
    y_hard = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    plain = loss_block(logits, y_hard, 0.0, 0.0, 0.0)
    focal = loss_block(logits, y_hard, 0.0, 0.0, 2.0)
    # focal ≈ 0.25 * plain when all p_t=0.5
    assert torch.allclose(focal, 0.25 * plain, atol=1e-5)


def test_focal_plus_smoothing_finite_and_positive():
    """Combined regime: both knobs on, loss should still be finite and > 0."""
    torch.manual_seed(0)
    logits = torch.randn(32) * 2.0
    y_hard = (torch.rand(32) > 0.5).float()
    loss = loss_block(logits, y_hard, eps_pos=0.05, eps_neg=0.02, gamma=2.0)
    assert torch.isfinite(loss)
    assert loss > 0.0


def test_gradient_flows_through_focal_weighted_path():
    """Focal weight is detached (no_grad), but gradient must still flow through BCE."""
    logits = torch.randn(16, requires_grad=True)
    y_hard = (torch.rand(16) > 0.5).float()
    loss = loss_block(logits, y_hard, 0.05, 0.02, 2.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0
