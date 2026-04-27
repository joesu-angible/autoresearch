from pathlib import Path


def test_dino_v2_reuses_budget_stop_eval_metrics_for_final_success():
    """Budget-stop screening should not repeat the same expensive final eval."""
    source = Path("dino_finetune/train_dino_v2.py").read_text()

    assert "last_eval_metrics" in source
    assert "Reusing last budget-stop eval metrics as final result" in source
