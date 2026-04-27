"""T11 verification: metrics_final_v2.json parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_loop.evaluators import parse_metrics


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_parse_minimal_required_keys(tmp_path: Path):
    p = tmp_path / "metrics_final_v2.json"
    _write(p, {"combined_metric": 0.86, "recall_at_1": 0.90, "mean_cosine": 0.81})
    out = parse_metrics(p)
    assert out["combined"] == 0.86
    assert out["recall_1"] == 0.90
    assert out["mean_cosine"] == 0.81
    assert out["recall_5"] == 0.90  # missing → recall@1 fallback


def test_parse_recall_at_5_none_falls_back_to_recall_at_1(tmp_path: Path):
    p = tmp_path / "metrics_final_v2.json"
    _write(p, {
        "combined_metric": 0.86,
        "recall_at_1": 0.90,
        "recall_at_5": None,
        "mean_cosine": 0.81,
    })
    out = parse_metrics(p)
    assert out["recall_5"] == 0.90


def test_parse_non_success_metrics_reports_clear_error(tmp_path: Path):
    p = tmp_path / "metrics_final_v2.json"
    _write(p, {"status": "failed", "error_type": "KeyError", "error": "recall@5"})
    with pytest.raises(ValueError, match="reports non-success status: failed.*recall@5"):
        parse_metrics(p)


def test_parse_passes_through_productness_keys(tmp_path: Path):
    p = tmp_path / "m.json"
    _write(p, {
        "combined_metric": 0.87, "recall_at_1": 0.91, "mean_cosine": 0.82,
        "recall_at_5": 0.94,
        "productness_loss": 0.12, "productness_acc": 0.95,
        "productness_pos_acc": 0.96, "productness_neg_acc": 0.94,
    })
    out = parse_metrics(p)
    assert out["recall_5"] == 0.94
    assert out["productness_loss"] == 0.12
    assert out["productness_pos_acc"] == 0.96


def test_missing_required_key_raises(tmp_path: Path):
    p = tmp_path / "m.json"
    _write(p, {"combined_metric": 0.86, "recall_at_1": 0.9})  # missing mean_cosine
    with pytest.raises(ValueError, match="missing required keys"):
        parse_metrics(p)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_metrics(tmp_path / "nope.json")
