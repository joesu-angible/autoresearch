"""Student V2 target adapter."""

from __future__ import annotations

from pathlib import Path

from research_loop.targets._base import TargetAdapter

REPO = Path(__file__).resolve().parents[2]

# Match train_v2.py's fallback: prefer /data when writable, else local workspace.
# Path is resolved at adapter-instantiation time so it always reads the same
# place the trainer just wrote to.
_DATA_OUT = Path("/data/training/reid/workspace/output/distill_final_lcnet050_v2/metrics_final_v2.json")
_LOCAL_OUT = REPO / "workspace" / "output" / "distill_final_lcnet050_v2" / "metrics_final_v2.json"


def _student_metrics_path() -> Path:
    """Pick whichever workspace base train_v2.py is using.

    Logic mirrors train_v2.py: /data is preferred when writable; otherwise
    fall back to the local repo workspace.
    """
    parent = _DATA_OUT.parent.parent  # /data/training/reid/workspace/output
    if parent.exists() and (parent.stat().st_mode & 0o200):
        return _DATA_OUT
    return _LOCAL_OUT


class StudentV2Target(TargetAdapter):
    name = "student_v2"
    REPO_DIR = REPO / "student_finetune"
    RESULTS_TSV = REPO / "student_finetune" / "results_v2.tsv"
    METRICS_JSON = _student_metrics_path()
    TRAIN_CMD = ["python", "train_v2.py"]
    DEFAULT_EPOCHS = 30
