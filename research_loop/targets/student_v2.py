"""Student V2 target adapter."""

from __future__ import annotations

from pathlib import Path

from research_loop.targets._base import TargetAdapter

REPO = Path(__file__).resolve().parents[2]


class StudentV2Target(TargetAdapter):
    name = "student_v2"
    REPO_DIR = REPO / "student_finetune"
    RESULTS_TSV = REPO / "student_finetune" / "results_v2.tsv"
    METRICS_JSON = (
        Path("/data/training/reid/workspace/output/distill_final_lcnet050_v2/metrics_final_v2.json")
    )
    TRAIN_CMD = ["python", "train_v2.py"]
