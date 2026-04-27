from pathlib import Path


def test_dino_v2_trainer_writes_metrics_where_adapter_reads_them():
    """The trainer runs from dino_finetune/, so output paths must not nest dino_finetune/dino_finetune."""
    source = Path("dino_finetune/train_dino_v2.py").read_text()

    assert 'ADAPTER_OUTPUT_DIR = "output/best_adapter"' in source
    assert 'LAST_ADAPTER_DIR = "output/last_adapter"' in source
    assert 'CHECKPOINT_PATH = "output/last_adapter/checkpoint.pt"' in source
    assert 'dino_finetune/output/best_adapter' not in source


def test_dino_v2_screening_does_not_resume_last_checkpoint_by_default():
    source = Path("dino_finetune/train_dino_v2.py").read_text()

    assert "RESUME_LAST_CHECKPOINT" in source
    assert 'os.environ.get("RESUME_LAST_CHECKPOINT", "0")' in source
    assert "has_adapter = RESUME_LAST_CHECKPOINT" in source
    assert "has_checkpoint = RESUME_LAST_CHECKPOINT" in source
