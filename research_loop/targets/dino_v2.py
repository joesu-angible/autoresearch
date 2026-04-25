"""DINO V2 target adapter."""

from __future__ import annotations

from pathlib import Path

from research_loop.targets._base import TargetAdapter

REPO = Path(__file__).resolve().parents[2]


class DinoV2Target(TargetAdapter):
    name = "dino_v2"
    REPO_DIR = REPO / "dino_finetune"
    RESULTS_TSV = REPO / "dino_finetune" / "results_v2.tsv"
    METRICS_JSON = REPO / "dino_finetune" / "output" / "metrics_final_v2.json"
    TRAIN_CMD = ["python", "train_dino_v2.py"]
