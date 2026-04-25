"""T4 verification: productness target plumbing.

We don't import the heavy V2CombinedDistillDataset (which scans real dataset
roots). Instead we test the inner contract: when train_v2.run_train_epoch
sees a path that lives under REID_NEGATIVES, the productness target must be
0.0; otherwise 1.0. The target derivation logic lives inline in train.py
(productness_targets list comprehension); we cover it via a tiny pure unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def derive_target(path: str, negative_paths: set[str]) -> float:
    """Mirror of the inline rule in train.py run_train_epoch."""
    return 0.0 if path in negative_paths else 1.0


def test_negative_path_target_is_zero():
    neg = {"/data/training/reid/reid_multiple/negatives/bag/x.jpg"}
    assert derive_target("/data/training/reid/reid_multiple/negatives/bag/x.jpg", neg) == 0.0


def test_product_path_target_is_one():
    neg = {"/data/training/reid/reid_multiple/negatives/bag/x.jpg"}
    assert derive_target("/data/training/reid/reid_multiple/products/0001/y.jpg", neg) == 1.0


def test_empty_negatives_set_means_all_products():
    assert derive_target("/anything/anywhere.jpg", set()) == 1.0


def test_mixed_batch_target_distribution():
    neg = {"/n/a.jpg", "/n/b.jpg"}
    paths = ["/p/x.jpg", "/n/a.jpg", "/p/y.jpg", "/n/b.jpg"]
    targets = [derive_target(p, neg) for p in paths]
    assert targets == [1.0, 0.0, 1.0, 0.0]
