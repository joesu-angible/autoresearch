"""Parse trainer outputs into objective scores for the tournament promote step.

Reads `metrics_final_v2.json` (student/dino V2) and emits a normalized dict.
Productness keys are passed through when present; absence is fine.
"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = ("combined_metric", "recall_at_1", "mean_cosine")


def parse_metrics(path: Path) -> dict[str, float]:
    """Load metrics_final_v2.json. Raises if required retrieval keys are missing."""
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    raw = json.loads(path.read_text())
    status = raw.get("status")
    if status and status != "success" and status != "in_progress":
        err = raw.get("error") or raw.get("error_type") or ""
        raise ValueError(f"metrics file {path} reports non-success status: {status}: {err}")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"metrics file {path} missing required keys: {missing}")
    recall_at_1 = float(raw["recall_at_1"])
    recall_at_5_raw = raw.get("recall_at_5")
    recall_at_5 = recall_at_1 if recall_at_5_raw is None else float(recall_at_5_raw)
    out: dict[str, float] = {
        "combined": float(raw["combined_metric"]),
        "recall_1": recall_at_1,
        "recall_5": recall_at_5,
        "mean_cosine": float(raw["mean_cosine"]),
    }
    # Productness keys are optional — pass through if present
    for k in ("productness_loss", "productness_acc", "productness_pos_acc", "productness_neg_acc"):
        if k in raw:
            out[k] = float(raw[k])
    return out
