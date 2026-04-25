"""T1 verification: metrics_progress_v2.json atomic write contract.

We don't run a full training cycle (that's a GPU run). We test the pure
helper function and its atomic-rename behavior in a tmp dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_v2 import write_metrics_progress  # noqa: E402


REQUIRED_KEYS = {
    "status", "version", "is_partial", "epochs_completed", "max_epochs",
    "combined_metric", "best_combined", "recall_at_1", "recall_at_5",
    "mean_cosine", "distill_loss",
    "productness_train_loss", "productness_train_acc",
    "productness_val_loss", "productness_val_acc",
    "productness_pos_acc", "productness_neg_acc",
}


def _full_kwargs():
    return dict(
        epoch=4, max_epochs=30,
        combined=0.79, recall_at_1=0.89, recall_at_5=0.94,
        mean_cosine=0.69, best_combined=0.79, distill_loss=0.31,
        productness_train_loss=0.08, productness_train_acc=0.96,
        productness_val_loss=0.45, productness_val_acc=0.84,
        productness_pos_acc=0.99, productness_neg_acc=0.81,
    )


def test_writes_valid_json_with_required_schema(tmp_path: Path):
    write_metrics_progress(tmp_path, **_full_kwargs())
    p = tmp_path / "metrics_progress_v2.json"
    assert p.exists()
    payload = json.loads(p.read_text())
    assert set(payload.keys()) >= REQUIRED_KEYS
    assert payload["is_partial"] is True
    assert payload["status"] == "in_progress"


def test_epochs_completed_is_one_indexed(tmp_path: Path):
    """epoch=4 (0-indexed loop counter) → epochs_completed=5 (human readable)."""
    kw = _full_kwargs()
    kw["epoch"] = 4
    write_metrics_progress(tmp_path, **kw)
    payload = json.loads((tmp_path / "metrics_progress_v2.json").read_text())
    assert payload["epochs_completed"] == 5


def test_atomic_replace_overwrites_existing(tmp_path: Path):
    """Second write replaces first — atomic via temp+rename."""
    kw = _full_kwargs()
    write_metrics_progress(tmp_path, **kw)
    kw["combined"] = 0.83
    kw["epoch"] = 9
    write_metrics_progress(tmp_path, **kw)
    payload = json.loads((tmp_path / "metrics_progress_v2.json").read_text())
    assert payload["combined_metric"] == 0.83
    assert payload["epochs_completed"] == 10


def test_no_tmp_file_left_after_successful_write(tmp_path: Path):
    """Atomic rename should leave no .tmp file on disk."""
    write_metrics_progress(tmp_path, **_full_kwargs())
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_all_numeric_values_serialize_as_json_numbers(tmp_path: Path):
    """Floats must serialize as JSON numbers, not strings."""
    write_metrics_progress(tmp_path, **_full_kwargs())
    raw = (tmp_path / "metrics_progress_v2.json").read_text()
    # Quick sanity: combined_metric must NOT be quoted
    assert '"combined_metric": 0.79' in raw
    assert '"is_partial": true' in raw  # JSON true, not Python True
