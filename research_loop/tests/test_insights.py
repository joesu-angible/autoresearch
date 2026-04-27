from __future__ import annotations

import csv
from pathlib import Path

from research_loop.candidate import (
    Candidate,
    CritiqueRecord,
    Decision,
    Outcome,
    OutcomeStartedRecord,
    PatchProposalRecord,
    SynthesisRecord,
    append_history,
)
from research_loop.insights import write_round_insight


DIFF = """diff --git a/student_finetune/train_v2.py b/student_finetune/train_v2.py
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -1 +1 @@
-LOSS_WEIGHT = 0.02
+LOSS_WEIGHT = 0.05
"""


def _candidate(kind: str, cid: str, rid: str, patch: str = "") -> Candidate:
    return Candidate(
        id=cid,
        kind=kind,  # type: ignore[arg-type]
        target="student_v2",
        round_id=rid,
        hypothesis=f"{kind} hypothesis explains what this candidate tried",
        expected_metric="combined +0.003",
        changed_files=["student_finetune/train_v2.py"] if patch else [],
        risks=["may regress recall"],
        rollback="rollback if combined regresses" if kind != "A" else "N/A — incumbent baseline",
        patch=patch,
    )


def test_write_round_insight_creates_result_like_tsv_and_markdown(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    run_dir = tmp_path / "runs" / "run-test"
    rid = "r1"

    a = _candidate("A", "cand-a", rid)
    b = _candidate("B", "cand-b", rid, DIFF)
    ab = _candidate("AB", "cand-ab", rid, DIFF)
    for c in (a, b, ab):
        append_history(history, c)
        append_history(history, OutcomeStartedRecord(
            candidate_id=c.id, round_id=rid, target="student_v2",
            pass_index=1, kind=c.kind, run_id="run-test",
        ))

    append_history(history, CritiqueRecord(
        round_id=rid, target="student_v2", pass_index=1,
        summary="Critic says productness is distracting retrieval.",
        problems=["productness over-weighted", "recall plateau"],
        raw="full critic text",
    ))
    append_history(history, PatchProposalRecord(
        round_id=rid, target="student_v2", pass_index=1, candidate_id=b.id,
        rationale="Author reduced productness pressure.", diff=DIFF, raw="raw author",
    ))
    append_history(history, SynthesisRecord(
        round_id=rid, target="student_v2", pass_index=1, candidate_id=ab.id,
        rationale="Synthesizer kept half the change.", diff=DIFF, raw="raw synth",
    ))
    append_history(history, Outcome(
        candidate_id=a.id, round_id=rid, target="student_v2", status="success",
        metrics={"combined": 0.660, "recall_1": 0.840, "mean_cosine": 0.480},
        elapsed_seconds=700.0, log_path="logs/a.log", metrics_json_path="metrics/a.json",
    ))
    append_history(history, Outcome(
        candidate_id=b.id, round_id=rid, target="student_v2", status="failed",
        metrics={"error": "CUDA out of memory"},
        elapsed_seconds=12.0, log_path="logs/b.log", metrics_json_path="metrics/b.json",
    ))
    append_history(history, Outcome(
        candidate_id=ab.id, round_id=rid, target="student_v2", status="success",
        metrics={"combined": 0.664, "recall_1": 0.846, "mean_cosine": 0.482},
        elapsed_seconds=710.0, log_path="logs/ab.log", metrics_json_path="metrics/ab.json",
    ))
    append_history(history, Decision(
        round_id=rid, target="student_v2", winner_id=ab.id, winner_kind="AB",
        promote=True, reason="AB beats A by more than noise band", deployable=False,
        deploy_failures=("combined below deploy threshold",),
    ))

    paths = write_round_insight(history, run_dir, run_id="run-test", target="student_v2", round_id=rid)

    tsv_path = run_dir / "autoreason_results.tsv"
    assert paths["tsv"] == tsv_path
    rows = list(csv.DictReader(tsv_path.read_text().splitlines(), delimiter="\t"))
    assert [r["kind"] for r in rows] == ["A", "B", "AB"]
    assert rows[0]["combined"] == "0.660000"
    assert rows[1]["status"] == "failed"
    assert rows[1]["failure_summary"] == "CUDA out of memory"
    assert rows[2]["winner"] == "yes"
    assert rows[2]["decision_reason"] == "AB beats A by more than noise band"
    assert rows[2]["synthesis_rationale"] == "Synthesizer kept half the change."

    summary = (run_dir / "rounds" / rid / "summary.md").read_text()
    assert "Critic says productness is distracting retrieval." in summary
    assert "Author reduced productness pressure." in summary
    assert "Synthesizer kept half the change." in summary
    assert "cand-b" in summary and "CUDA out of memory" in summary
    assert "Winner: AB cand-ab" in summary

    latest = run_dir / "latest_round.md"
    assert latest.exists()
    assert latest.read_text() == summary
