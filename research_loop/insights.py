"""Human-readable autoreason insight reports.

The JSONL history is complete but painful to read. This module materializes a
result-like TSV plus per-round Markdown so humans and future AI agents can learn
what each round tried without reconstructing many record types by hand.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from research_loop.candidate import (
    Candidate,
    CritiqueRecord,
    Decision,
    Outcome,
    OutcomeStartedRecord,
    PatchProposalRecord,
    SynthesisRecord,
    read_critiques,
    read_decisions,
    read_history,
    read_outcomes,
    read_outcomes_started,
    read_patch_proposals,
    read_syntheses,
)

TSV_FIELDS = [
    "run_id",
    "pass_index",
    "round_id",
    "target",
    "kind",
    "candidate_id",
    "status",
    "winner",
    "promote",
    "combined",
    "recall_1",
    "recall_5",
    "mean_cosine",
    "productness_neg_acc",
    "elapsed_seconds",
    "failure_summary",
    "hypothesis",
    "critic_summary",
    "author_rationale",
    "synthesis_rationale",
    "decision_reason",
    "log_path",
    "metrics_json_path",
]


def _short(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fmt_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def _failure_summary(outcome: Outcome | None) -> str:
    if outcome is None:
        return "not run"
    if outcome.status == "success":
        return ""
    metrics = outcome.metrics or {}
    for key in ("error", "failure", "failure_summary", "message", "stderr_tail"):
        value = metrics.get(key)
        if value:
            return _short(str(value), 300)
    if outcome.status == "failed":
        return "failed; inspect candidate log"
    return outcome.status


def _first_or_none(items: list[Any]) -> Any | None:
    return items[-1] if items else None


def _records_for_round(history_path: Path, round_id: str) -> tuple[
    list[Candidate], list[Outcome], list[Decision], CritiqueRecord | None,
    PatchProposalRecord | None, SynthesisRecord | None, list[OutcomeStartedRecord],
]:
    candidates = [c for c in read_history(history_path) if c.round_id == round_id]
    outcomes = list(read_outcomes(history_path, round_id=round_id))
    decisions = list(read_decisions(history_path, round_id=round_id))
    critiques = list(read_critiques(history_path, round_id=round_id))
    proposals = list(read_patch_proposals(history_path, round_id=round_id))
    syntheses = list(read_syntheses(history_path, round_id=round_id))
    starts = [s for s in read_outcomes_started(history_path) if s.round_id == round_id]
    return candidates, outcomes, decisions, _first_or_none(critiques), _first_or_none(proposals), _first_or_none(syntheses), starts


def _pass_index(starts: list[OutcomeStartedRecord]) -> str:
    values = [s.pass_index for s in starts if s.pass_index]
    return str(values[0]) if values else ""


def _candidate_rows(
    *,
    run_id: str,
    target: str,
    round_id: str,
    candidates: list[Candidate],
    outcomes: list[Outcome],
    decision: Decision | None,
    critique: CritiqueRecord | None,
    proposal: PatchProposalRecord | None,
    synthesis: SynthesisRecord | None,
    starts: list[OutcomeStartedRecord],
) -> list[dict[str, str]]:
    outcome_by_id = {o.candidate_id: o for o in outcomes}
    kind_order = {"A": 0, "B": 1, "AB": 2}
    rows: list[dict[str, str]] = []
    for candidate in sorted(candidates, key=lambda c: (kind_order.get(c.kind, 99), c.id)):
        outcome = outcome_by_id.get(candidate.id)
        metrics = outcome.metrics if outcome else {}
        row = {
            "run_id": run_id,
            "pass_index": _pass_index(starts),
            "round_id": round_id,
            "target": target,
            "kind": candidate.kind,
            "candidate_id": candidate.id,
            "status": outcome.status if outcome else "not_run",
            "winner": "yes" if decision and candidate.id == decision.winner_id else "",
            "promote": "yes" if decision and candidate.id == decision.winner_id and decision.promote else "",
            "combined": _fmt_metric(metrics.get("combined")),
            "recall_1": _fmt_metric(metrics.get("recall_1")),
            "recall_5": _fmt_metric(metrics.get("recall_5", metrics.get("recall_at_5"))),
            "mean_cosine": _fmt_metric(metrics.get("mean_cosine")),
            "productness_neg_acc": _fmt_metric(metrics.get("productness_neg_acc")),
            "elapsed_seconds": _fmt_metric(outcome.elapsed_seconds if outcome else None),
            "failure_summary": _failure_summary(outcome),
            "hypothesis": _short(candidate.hypothesis, 300),
            "critic_summary": _short(critique.summary if critique else "", 300),
            "author_rationale": _short(proposal.rationale if proposal and candidate.kind == "B" else "", 300),
            "synthesis_rationale": _short(synthesis.rationale if synthesis and candidate.kind == "AB" else "", 300),
            "decision_reason": _short(decision.reason if decision else "", 300),
            "log_path": outcome.log_path if outcome else "",
            "metrics_json_path": outcome.metrics_json_path if outcome else "",
        }
        rows.append(row)
    return rows


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _format_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| kind | candidate | status | combined | recall_1 | mean_cosine | elapsed | failure | log |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {kind} | {candidate_id} | {status} | {combined} | {recall_1} | "
            "{mean_cosine} | {elapsed_seconds} | {failure_summary} | {log_path} |".format(**r)
        )
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    *,
    run_id: str,
    round_id: str,
    target: str,
    critique: CritiqueRecord | None,
    proposal: PatchProposalRecord | None,
    synthesis: SynthesisRecord | None,
    decision: Decision | None,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = "none"
    if decision:
        winner = f"{decision.winner_kind} {decision.winner_id}"
    problems = ""
    if critique and critique.problems:
        problems = "\n".join(f"- {p}" for p in critique.problems)
    else:
        problems = "- n/a"
    text = f"""# Autoreason Round Insight

Run: {run_id}
Round: {round_id}
Target: {target}
Winner: {winner}
Promote: {"yes" if decision and decision.promote else "no"}
Decision: {_short(decision.reason if decision else "no decision yet", 500)}

## Critic
{_short(critique.summary if critique else "n/a", 1000)}

Problems:
{problems}

## Author B
{_short(proposal.rationale if proposal else "n/a", 1000)}

## Synthesizer AB
{_short(synthesis.rationale if synthesis else "n/a", 1000)}

## Outcomes
{_format_table(rows)}

## How to read this
- `autoreason_results.tsv` is the compact result-like ledger for the run.
- This Markdown file explains the round: what the critic believed, what B/AB tried, what happened, and why the winner was chosen.
"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_round_insight(
    history_path: Path,
    run_dir: Path,
    *,
    run_id: str,
    target: str,
    round_id: str,
) -> dict[str, Path]:
    """Write human-readable insight artifacts for one completed round.

    Returns paths for callers/tests. The TSV is rebuilt from all rounds associated
    with this run_id each time, so resume/retry does not duplicate rows.
    """
    candidates, outcomes, decisions, critique, proposal, synthesis, starts = _records_for_round(history_path, round_id)
    decision = _first_or_none(decisions)
    rows = _candidate_rows(
        run_id=run_id, target=target, round_id=round_id,
        candidates=candidates, outcomes=outcomes, decision=decision,
        critique=critique, proposal=proposal, synthesis=synthesis, starts=starts,
    )

    # Rebuild the run-wide TSV from every round that has an OutcomeStartedRecord
    # for this run. Include the current round even if legacy history lacks starts.
    run_round_ids = []
    seen: set[str] = set()
    for s in read_outcomes_started(history_path, run_id=run_id):
        if s.round_id not in seen:
            seen.add(s.round_id)
            run_round_ids.append(s.round_id)
    if round_id not in seen:
        run_round_ids.append(round_id)

    all_rows: list[dict[str, str]] = []
    for rid in run_round_ids:
        cands, outs, decs, crit, prop, synth, st = _records_for_round(history_path, rid)
        all_rows.extend(_candidate_rows(
            run_id=run_id, target=target, round_id=rid,
            candidates=cands, outcomes=outs, decision=_first_or_none(decs),
            critique=crit, proposal=prop, synthesis=synth, starts=st,
        ))

    tsv_path = run_dir / "autoreason_results.tsv"
    md_path = run_dir / "rounds" / round_id / "summary.md"
    latest_path = run_dir / "latest_round.md"
    _write_tsv(tsv_path, all_rows)
    _write_markdown(
        md_path, run_id=run_id, round_id=round_id, target=target,
        critique=critique, proposal=proposal, synthesis=synthesis,
        decision=decision, rows=rows,
    )
    shutil.copyfile(md_path, latest_path)
    return {"tsv": tsv_path, "summary": md_path, "latest": latest_path}
