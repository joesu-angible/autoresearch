"""Autoreason tournament CLI — propose / rank / run / promote / run-round.

Subcommands:

  propose --target {student_v2,dino_v2} --hypothesis "..."
      Emits an A/B/AB triple to research_loop/history.jsonl, all stamped with
      a fresh round_id. A is the do-nothing incumbent; B/AB are placeholder
      patches the user fills in (or `--baseline-only` to skip them).

  rank [--round <id>]
      Reads outstanding candidates and prints judge ranking.

  run --candidate <id> [--epochs N] [--dry-run]
      Runs the named candidate end-to-end via its target adapter:
        adapter.apply_patch → adapter.train → parse metrics
        → write Outcome to history.jsonl
        → adapter.log_row → append to results_v2.tsv
      A real run blocks for the full training duration (hours). Use tmux/nohup.

  promote --round <id>
      Gathers Outcomes for the round, calls promote.decide(), writes a
      Decision record. Reports the deployment-gate verdict separately.

  run-round --target ... --hypothesis "..." [--epochs N] [--baseline-only]
      One-shot: propose → rank → run all → promote. Use this for the
      production baseline (--baseline-only runs only A) and any subsequent
      end-to-end experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_loop.candidate import (
    Candidate,
    Decision,
    Outcome,
    append_history,
    find_candidate,
    new_round_id,
    read_history,
    read_outcomes,
)
from research_loop.judges import rank as rank_candidates
from research_loop.promote import (
    CandidateResult,
    decide,
    is_deployable,
)
from research_loop.targets import DinoV2Target, StudentV2Target

HISTORY_PATH = Path(__file__).resolve().parent / "history.jsonl"

TARGETS: dict[str, type] = {
    "student_v2": StudentV2Target,
    "dino_v2": DinoV2Target,
}


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

def _make_baseline_a(target: str, round_id: str, hypothesis: str) -> Candidate:
    return Candidate(
        kind="A",
        target=target,
        round_id=round_id,
        hypothesis=hypothesis or "do nothing — keep the current best of results_v2.tsv",
        expected_metric="combined Δ +0.000 (incumbent baseline)",
        changed_files=[],
        risks=[],
        rollback="N/A — incumbent baseline",
        patch="",
    )


def _make_placeholder_b(target: str, round_id: str, hypothesis: str, parent_id: str) -> Candidate:
    target_file = (
        "student_finetune/train_v2.py" if target == "student_v2"
        else "dino_finetune/train_dino_v2.py"
    )
    return Candidate(
        kind="B",
        target=target,
        round_id=round_id,
        hypothesis=hypothesis or "TODO: fill in concrete hypothesis",
        expected_metric="combined +0.005 (TODO: replace)",
        changed_files=[target_file],
        risks=["may regress recall@1"],
        rollback="combined < incumbent - 0.005",
        patch="--- a/placeholder\n+++ b/placeholder\n@@\n+# TODO: real patch\n",
        parent_incumbent_id=parent_id,
    )


def _make_placeholder_ab(target: str, round_id: str, b: Candidate, parent_id: str) -> Candidate:
    return Candidate(
        kind="AB",
        target=target,
        round_id=round_id,
        hypothesis="conservative synthesis — apply B at half strength",
        expected_metric="combined +0.002 (TODO)",
        changed_files=b.changed_files,
        risks=["may underdeliver"],
        rollback=b.rollback,
        patch="--- a/placeholder\n+++ b/placeholder\n@@\n+# TODO: synthesis patch\n",
        parent_incumbent_id=parent_id,
    )


def cmd_propose(target: str, hypothesis: str, baseline_only: bool, round_id: str | None = None) -> int:
    if target not in TARGETS:
        print(f"Unknown target: {target}", file=sys.stderr)
        return 2
    rid = round_id or new_round_id()
    a = _make_baseline_a(target, rid, hypothesis)
    append_history(HISTORY_PATH, a)
    print(f"Round {rid} proposed for target={target}:")
    print(f"  A  id={a.id}  {a.hypothesis[:60]}")
    if baseline_only:
        return 0
    b = _make_placeholder_b(target, rid, hypothesis, parent_id=a.id)
    ab = _make_placeholder_ab(target, rid, b, parent_id=a.id)
    append_history(HISTORY_PATH, b)
    append_history(HISTORY_PATH, ab)
    print(f"  B  id={b.id}  {b.hypothesis[:60]}")
    print(f"  AB id={ab.id}  {ab.hypothesis[:60]}")
    return 0


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def cmd_rank(round_id: str | None) -> int:
    candidates = list(read_history(HISTORY_PATH))
    if round_id:
        candidates = [c for c in candidates if c.round_id == round_id]
    if not candidates:
        print(f"No candidates{' for round ' + round_id if round_id else ''}.")
        return 0
    ordering = rank_candidates(candidates)
    print(f"Ranking ({len(ordering)} candidates):")
    for o in ordering:
        print(f"  score={o.score:.3f}  kind={o.kind}  id={o.candidate_id}")
    return 0


# ---------------------------------------------------------------------------
# run — actual subprocess + outcome record + tsv row
# ---------------------------------------------------------------------------

def _run_one(candidate: Candidate, epochs: int | None, dry_run: bool) -> Outcome:
    target_cls = TARGETS[candidate.target]
    adapter = target_cls()
    adapter.apply_patch(candidate)  # validates V1-safety; raises if bad
    train_outcome = adapter.train(candidate, max_epochs=epochs, dry_run=dry_run)

    outcome = Outcome(
        candidate_id=candidate.id,
        round_id=candidate.round_id,
        target=candidate.target,
        status=train_outcome.status,
        metrics=train_outcome.metrics,
        elapsed_seconds=train_outcome.elapsed_seconds,
        log_path=str(train_outcome.log_path),
        metrics_json_path=str(train_outcome.metrics_json_path),
    )
    append_history(HISTORY_PATH, outcome)
    if not dry_run and train_outcome.status == "success":
        adapter.log_row(candidate, train_outcome)
    return outcome


def cmd_run(candidate_id: str, epochs: int | None, dry_run: bool) -> int:
    candidate = find_candidate(HISTORY_PATH, candidate_id)
    if candidate is None:
        print(f"Candidate not found: {candidate_id}", file=sys.stderr)
        return 2
    print(f"Running candidate {candidate.id} (kind={candidate.kind}, target={candidate.target}, round={candidate.round_id})")
    if dry_run:
        print("  [dry-run]")
    outcome = _run_one(candidate, epochs, dry_run)
    print(f"  status={outcome.status}  elapsed={outcome.elapsed_seconds:.1f}s")
    if outcome.metrics:
        print(f"  combined={outcome.metrics.get('combined', 0.0):.4f}  recall@1={outcome.metrics.get('recall_1', 0.0):.4f}  neg_acc={outcome.metrics.get('productness_neg_acc', 0.0):.4f}")
    return 0 if outcome.status == "success" or dry_run else 1


# ---------------------------------------------------------------------------
# promote — apply guardrails over a round's outcomes, write Decision
# ---------------------------------------------------------------------------

def _outcome_to_candidate_result(outcome: Outcome, candidate: Candidate) -> CandidateResult:
    m = outcome.metrics
    return CandidateResult(
        candidate_id=outcome.candidate_id,
        kind=candidate.kind,
        combined=m.get("combined", 0.0),
        recall_1=m.get("recall_1", 0.0),
        mean_cosine=m.get("mean_cosine", 0.0),
        productness_pos_acc=m.get("productness_pos_acc"),
        productness_neg_acc=m.get("productness_neg_acc"),
        has_rollback=bool(candidate.rollback.strip())
            and candidate.rollback.strip().lower() != "n/a"
            or candidate.kind == "A",
    )


def cmd_promote(round_id: str) -> int:
    candidates = {c.id: c for c in read_history(HISTORY_PATH) if c.round_id == round_id}
    if not candidates:
        print(f"No candidates for round {round_id}", file=sys.stderr)
        return 2
    outcomes = [o for o in read_outcomes(HISTORY_PATH, round_id=round_id)
                if o.status == "success"]
    if not outcomes:
        print(f"No successful outcomes for round {round_id}", file=sys.stderr)
        return 1

    results = [
        _outcome_to_candidate_result(o, candidates[o.candidate_id])
        for o in outcomes if o.candidate_id in candidates
    ]
    if not any(r.kind == "A" for r in results):
        print(f"Round {round_id} has no A outcome — cannot promote", file=sys.stderr)
        return 1

    decision = decide(results)
    winner_result = next((r for r in results if r.candidate_id == decision.winner_id), None)
    deploy_verdict = is_deployable(winner_result) if winner_result else None
    target = candidates[decision.winner_id].target

    decision_record = Decision(
        round_id=round_id,
        target=target,
        winner_id=decision.winner_id,
        winner_kind=decision.winner_kind,
        promote=decision.promote,
        reason=decision.reason,
        deployable=bool(deploy_verdict.deployable) if deploy_verdict else False,
        deploy_failures=tuple(deploy_verdict.reasons) if deploy_verdict else (),
    )
    append_history(HISTORY_PATH, decision_record)

    print(f"Round {round_id} decision:")
    print(f"  winner: {decision.winner_kind}  id={decision.winner_id}")
    print(f"  promote: {decision.promote}")
    print(f"  reason: {decision.reason}")
    print(f"  deployable: {decision_record.deployable}")
    if not decision_record.deployable:
        for r in decision_record.deploy_failures:
            print(f"    - {r}")
    return 0


# ---------------------------------------------------------------------------
# run-round — one-shot orchestrator
# ---------------------------------------------------------------------------

def cmd_run_round(target: str, hypothesis: str, epochs: int | None,
                  baseline_only: bool, dry_run: bool) -> int:
    rid = new_round_id()
    rc = cmd_propose(target, hypothesis, baseline_only=baseline_only, round_id=rid)
    if rc != 0:
        return rc
    cmd_rank(rid)

    round_candidates = [c for c in read_history(HISTORY_PATH) if c.round_id == rid]
    print(f"\nExecuting {len(round_candidates)} candidate(s) for round {rid}")
    for c in round_candidates:
        print(f"\n=== running {c.kind} ({c.id}) ===")
        outcome = _run_one(c, epochs, dry_run)
        print(f"    status={outcome.status}")
        if outcome.status != "success" and not dry_run:
            print(f"    aborting round — candidate {c.kind} failed", file=sys.stderr)
            return 1

    if dry_run:
        print(f"\n[dry-run] skipping promote for round {rid}")
        return 0
    return cmd_promote(rid)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_loop.tournament")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_propose = sub.add_parser("propose", help="emit A/B/AB candidates for a new round")
    p_propose.add_argument("--target", required=True, choices=list(TARGETS))
    p_propose.add_argument("--hypothesis", default="")
    p_propose.add_argument("--baseline-only", action="store_true",
                           help="emit only A (incumbent); skip B/AB placeholders")

    p_rank = sub.add_parser("rank", help="judge-rank candidates")
    p_rank.add_argument("--round", default=None, help="filter by round_id")

    p_run = sub.add_parser("run", help="execute one candidate end-to-end")
    p_run.add_argument("--candidate", required=True)
    p_run.add_argument("--epochs", type=int, default=None,
                       help="override target's DEFAULT_EPOCHS")
    p_run.add_argument("--dry-run", action="store_true",
                       help="skip subprocess; record a noop outcome")

    p_promote = sub.add_parser("promote", help="apply guardrails + write Decision for a round")
    p_promote.add_argument("--round", required=True)

    p_round = sub.add_parser("run-round", help="propose + rank + run all + promote")
    p_round.add_argument("--target", required=True, choices=list(TARGETS))
    p_round.add_argument("--hypothesis", default="")
    p_round.add_argument("--epochs", type=int, default=None)
    p_round.add_argument("--baseline-only", action="store_true",
                         help="run only the A (incumbent baseline); useful for first run")
    p_round.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "propose":
        return cmd_propose(args.target, args.hypothesis, baseline_only=args.baseline_only)
    if args.cmd == "rank":
        return cmd_rank(args.round)
    if args.cmd == "run":
        return cmd_run(args.candidate, args.epochs, args.dry_run)
    if args.cmd == "promote":
        return cmd_promote(args.round)
    if args.cmd == "run-round":
        return cmd_run_round(args.target, args.hypothesis, args.epochs,
                             baseline_only=args.baseline_only, dry_run=args.dry_run)
    return 2


if __name__ == "__main__":
    sys.exit(main())
