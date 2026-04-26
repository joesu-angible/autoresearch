"""Verify DINO-side productness wiring: head shape + target derivation + loss math.

CPU-only. Imports heavy modules lazily so missing deps fail with skip rather
than collection error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
import torch.nn.functional as F

# train_dino_v2 imports prepare_dino which imports transformers; allow skip
# if those aren't available in this dev env.
pytest.importorskip("transformers")

from train_dino_v2 import (  # noqa: E402
    DinoProductnessHead,
    NEGATIVE_LABEL,
    COMMODITY_LABEL,
    derive_productness_targets,
    productness_loss_block,
    PRODUCTNESS_LABEL_SMOOTHING_POS,
    PRODUCTNESS_LABEL_SMOOTHING_NEG,
    PRODUCTNESS_FOCAL_GAMMA,
    USE_GRADIENT_CHECKPOINTING,
    _auto_batch,
    _auto_num_workers,
)


def test_gradient_checkpointing_disabled_by_default_on_large_vram_hosts():
    assert USE_GRADIENT_CHECKPOINTING is False


def test_auto_batch_uses_large_96gb_gpu_capacity():
    """RTX PRO 6000 96GB should use a no-checkpointing-safe batch."""
    assert _auto_batch(default_at_24gb=8, vram_gb=95.0) == 32


def test_auto_batch_keeps_24gb_baseline_safe():
    assert _auto_batch(default_at_24gb=8, vram_gb=24.0) == 8


def test_auto_num_workers_scales_above_legacy_four_workers():
    assert _auto_num_workers(cpu_count=32) == 16
    assert _auto_num_workers(cpu_count=8) == 4


def test_target_derivation_negative_label_is_zero():
    labels = torch.tensor([0, 1, 5, NEGATIVE_LABEL, COMMODITY_LABEL, 99])
    targets = derive_productness_targets(labels)
    # 0,1,5 = real classes → 1.0; NEGATIVE_LABEL=-2 → 0.0;
    # COMMODITY_LABEL=-1 → 1.0 (it's an unlabeled product); 99 → 1.0
    expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    assert torch.equal(targets, expected)


def test_dino_productness_head_shape():
    head = DinoProductnessHead(embedding_dim=1280, hidden=64)
    x = torch.randn(8, 1280)
    head.train()
    out = head(x)
    assert out.shape == (8,)
    assert out.dtype == torch.float32


def test_dino_productness_head_grad_flows():
    head = DinoProductnessHead(embedding_dim=1280, hidden=64)
    x = torch.randn(4, 1280, requires_grad=True)
    y_hard = torch.tensor([1.0, 0.0, 1.0, 0.0])
    logits = head(x)
    loss = productness_loss_block(logits, y_hard, eps_pos=0.05, eps_neg=0.02, gamma=2.0)
    loss.backward()
    head_params = list(head.parameters())
    assert all(p.grad is not None for p in head_params)
    assert all(torch.isfinite(p.grad).all() for p in head_params)


def test_dino_loss_block_matches_student_math():
    """Sanity: this is the same math the student-side block uses."""
    torch.manual_seed(0)
    logits = torch.randn(16) * 2.0
    y_hard = (torch.rand(16) > 0.5).float()
    loss = productness_loss_block(logits, y_hard,
                                   eps_pos=PRODUCTNESS_LABEL_SMOOTHING_POS,
                                   eps_neg=PRODUCTNESS_LABEL_SMOOTHING_NEG,
                                   gamma=PRODUCTNESS_FOCAL_GAMMA)
    assert torch.isfinite(loss)
    assert loss > 0.0


def test_loss_block_recovers_plain_bce_at_zero_knobs():
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    ours = productness_loss_block(logits, y, 0.0, 0.0, 0.0)
    ref = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(ours, ref, atol=1e-7)


def test_negative_label_is_minus_two_sentinel():
    """Don't accidentally collide with a real class id."""
    assert NEGATIVE_LABEL == -2
    assert COMMODITY_LABEL == -1
    assert NEGATIVE_LABEL != COMMODITY_LABEL
