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
    CritiqueRecord,
    Decision,
    Outcome,
    PatchProposalRecord,
    SynthesisRecord,
    append_history,
    find_candidate,
    new_round_id,
    read_history,
    read_outcomes,
)
from research_loop.judges import rank as rank_candidates
from research_loop.patch import apply_patch
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
        status=outcome.status,
    )


def cmd_promote(round_id: str) -> int:
    candidates = {c.id: c for c in read_history(HISTORY_PATH) if c.round_id == round_id}
    if not candidates:
        print(f"No candidates for round {round_id}", file=sys.stderr)
        return 2
    outcomes = list(read_outcomes(HISTORY_PATH, round_id=round_id))
    if not outcomes:
        print(f"No outcomes recorded for round {round_id}", file=sys.stderr)
        return 1
    # Pass all outcomes (including timeouts) to decide() so the rejection
    # reasons are visible in the Decision.reason field, not silently filtered.

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
# autoreason — fully autonomous LLM-driven loop
# ---------------------------------------------------------------------------

# Trainer source paths each target's Author B agent should patch.
_TRAINER_PATHS = {
    "student_v2": "student_finetune/train_v2.py",
    "dino_v2":    "dino_finetune/train_dino_v2.py",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_trainer(target: str) -> tuple[str, str]:
    """Return (trainer_source, trainer_repo_relative_path) for the target."""
    rel = _TRAINER_PATHS[target]
    return (REPO_ROOT / rel).read_text(), rel


def _read_results_tail(target: str, n_rows: int = 30) -> str:
    """Last N rows of the target's results_v2.tsv (header + tail), or '' if missing."""
    path = TARGETS[target]().RESULTS_TSV
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    if not lines:
        return ""
    header = lines[0]
    tail = lines[-n_rows:]
    if tail and tail[0] == header:
        return "\n".join(tail)
    return "\n".join([header] + tail)


def _read_recent_outcomes_jsonl(target: str, n: int = 10) -> str:
    """Last N outcome JSONL records for the target, newest-first becomes oldest-first."""
    relevant = [o for o in read_outcomes(HISTORY_PATH) if o.target == target]
    return "\n".join(o.to_jsonl() for o in relevant[-n:])


def _git_status_clean(repo: Path) -> bool:
    """Return True iff `git status --porcelain` is empty in `repo`."""
    import subprocess
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return res.returncode == 0 and not res.stdout.strip()


def cmd_autoreason(
    target: str,
    *,
    max_passes: int,
    convergence: int,
    max_seconds_per_candidate: float | None,
    hypothesis_seed: str,
    dry_run: bool,
    llm_cli: str | None = None,
    llm_model: str | None = None,
    llm_provider: str = "auto",
    agent_client_factory=None,
) -> int:
    """Fully autonomous autoreason loop.

    Each pass: fresh Critic → fresh Author B → fresh Synthesizer → run A/B/AB
    via the target adapter (under per-candidate time budget) → promote.decide().
    Winner becomes the new A; loop terminates after `convergence` consecutive
    A wins or after `max_passes` total passes.

    LLM access is via subprocess to a local CLI (hermes / claude / codex).
    `llm_cli` defaults to AUTORESEARCH_LLM_CLI env or "hermes". `llm_model`
    is passed through to whichever CLI is selected.

    `agent_client_factory` is injected for testing — production callers pass
    None and we construct a real CLI-backed client.
    """
    if target not in TARGETS:
        print(f"Unknown target: {target}", file=sys.stderr)
        return 2

    if not dry_run and not _git_status_clean(REPO_ROOT):
        print(
            "Working tree is dirty. autoreason will apply and revert patches "
            "via `git apply`; commit or stash uncommitted changes before running.",
            file=sys.stderr,
        )
        return 2

    if agent_client_factory is None:
        from research_loop.agents.client import make_llm_client
        agent_client_factory = lambda: make_llm_client(
            llm_cli, model=llm_model, provider=llm_provider,
        )
    client = agent_client_factory()
    print(f"  LLM client: {getattr(client, 'name', type(client).__name__)}"
          + (f" model={llm_model}" if llm_model else ""))

    from research_loop.agents.author import AuthorBAgent
    from research_loop.agents.critic import CriticAgent
    from research_loop.agents.synthesizer import SynthesizerAgent

    critic = CriticAgent(client)
    author = AuthorBAgent(client)
    synthesizer = SynthesizerAgent(client)

    consecutive_a_wins = 0
    last_round_id: str | None = None

    for pass_index in range(1, max_passes + 1):
        print(f"\n=== autoreason pass {pass_index}/{max_passes} ===")
        round_id = new_round_id()

        trainer_source, trainer_path = _read_trainer(target)
        results_tail = _read_results_tail(target)
        recent_outcomes = _read_recent_outcomes_jsonl(target)

        # 1. Critic
        critique = critic.critique(
            trainer_source=trainer_source,
            results_tsv_tail=results_tail,
            recent_outcomes_json=recent_outcomes,
        )
        append_history(HISTORY_PATH, CritiqueRecord(
            round_id=round_id, target=target, pass_index=pass_index,
            summary=critique.summary, problems=critique.problems, raw=critique.raw,
        ))
        print(f"  critic: {len(critique.problems)} problem(s) found")

        # 2. Author B
        b_proposal = author.author(
            trainer_source=trainer_source,
            trainer_path=trainer_path,
            critique_text=critique.raw,
        )

        # 3. Synthesizer (sees only patches, anonymized labels)
        ab_synthesis = synthesizer.synthesize(
            patch_x="", patch_y=b_proposal.diff,  # X=A=empty (do-nothing); Y=B
            trainer_path=trainer_path,
        )

        # 4. Build A / B / AB Candidate records for this round
        a_candidate = _make_baseline_a(target, round_id, hypothesis_seed)
        b_candidate = Candidate(
            kind="B", target=target, round_id=round_id,
            hypothesis=critique.summary[:200] or "Address critic findings",
            expected_metric="combined Δ > 0.003 (TBD by run)",
            changed_files=[trainer_path] if b_proposal.diff else [],
            risks=critique.problems[:3] or ["LLM-generated patch — unverified"],
            rollback="combined regresses below incumbent by > 0.005",
            patch=b_proposal.diff or "--- a/_noop\n+++ b/_noop\n",
            parent_incumbent_id=a_candidate.id,
        ) if b_proposal.diff else None
        ab_candidate = Candidate(
            kind="AB", target=target, round_id=round_id,
            hypothesis="Conservative synthesis of A and B",
            expected_metric="combined Δ > 0.003 (TBD by run)",
            changed_files=[trainer_path] if ab_synthesis.diff else [],
            risks=["synthesis-only"],
            rollback="combined regresses below incumbent by > 0.005",
            patch=ab_synthesis.diff or "--- a/_noop\n+++ b/_noop\n",
            parent_incumbent_id=a_candidate.id,
        ) if ab_synthesis.diff else None

        for c in (a_candidate, b_candidate, ab_candidate):
            if c is not None:
                append_history(HISTORY_PATH, c)

        # Audit-trail records for the LLM calls
        if b_candidate is not None:
            append_history(HISTORY_PATH, PatchProposalRecord(
                round_id=round_id, target=target, pass_index=pass_index,
                candidate_id=b_candidate.id,
                rationale=b_proposal.rationale, diff=b_proposal.diff,
                raw=b_proposal.raw,
            ))
        if ab_candidate is not None:
            append_history(HISTORY_PATH, SynthesisRecord(
                round_id=round_id, target=target, pass_index=pass_index,
                candidate_id=ab_candidate.id,
                rationale=ab_synthesis.rationale, diff=ab_synthesis.diff,
                raw=ab_synthesis.raw,
            ))

        # 5. Run A / B / AB. A is do-nothing — we still need its outcome
        # (first pass: real training run; later passes: reuse last A's outcome).
        candidates_to_run = [a_candidate]
        if b_candidate is not None:
            candidates_to_run.append(b_candidate)
        if ab_candidate is not None:
            candidates_to_run.append(ab_candidate)

        for cand in candidates_to_run:
            print(f"  running {cand.kind} ({cand.id})")
            adapter = TARGETS[target]()
            adapter.apply_patch(cand)  # validates V1-safety; no-op for A
            if cand.kind == "A" or not cand.patch.strip():
                # Empty patch = no working-tree change
                outcome = adapter.train(cand, max_seconds=max_seconds_per_candidate, dry_run=dry_run)
            else:
                with apply_patch(cand.patch, repo=REPO_ROOT, require_clean_tree=False):
                    outcome = adapter.train(cand, max_seconds=max_seconds_per_candidate, dry_run=dry_run)
            outcome_record = Outcome(
                candidate_id=cand.id, round_id=round_id, target=target,
                status=outcome.status, metrics=outcome.metrics,
                elapsed_seconds=outcome.elapsed_seconds,
                log_path=str(outcome.log_path),
                metrics_json_path=str(outcome.metrics_json_path),
            )
            append_history(HISTORY_PATH, outcome_record)
            if not dry_run and outcome.status == "success":
                adapter.log_row(cand, outcome)
            print(f"    status={outcome.status}")

        # 6. Promote — A vs B vs AB under guardrails
        rc = cmd_promote(round_id)
        if rc != 0:
            print(f"  promote failed for round {round_id}", file=sys.stderr)
            return rc

        # 7. Convergence check
        decisions = [d for d in __import__("research_loop.candidate", fromlist=["read_decisions"]).read_decisions(HISTORY_PATH, round_id=round_id)]
        if not decisions:
            print(f"  no decision recorded for round {round_id}", file=sys.stderr)
            return 1
        decision = decisions[-1]
        if decision.winner_kind == "A":
            consecutive_a_wins += 1
        else:
            consecutive_a_wins = 0
        last_round_id = round_id
        print(f"  decision: winner={decision.winner_kind}, consecutive A={consecutive_a_wins}/{convergence}")

        if consecutive_a_wins >= convergence:
            print(f"\nConverged at pass {pass_index} (A won {convergence} consecutive rounds).")
            return 0

    print(f"\nMax passes ({max_passes}) reached without convergence. Last round: {last_round_id}")
    return 1


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

    p_auto = sub.add_parser("autoreason",
        help="fully-autonomous LLM-driven loop (Critic + Author B + Synthesizer)")
    p_auto.add_argument("--target", required=True, choices=list(TARGETS))
    p_auto.add_argument("--max-passes", type=int, default=15,
                        help="ceiling on number of refinement passes (default 15)")
    p_auto.add_argument("--convergence", type=int, default=2,
                        help="terminate after this many consecutive A wins (default 2)")
    p_auto.add_argument("--max-seconds-per-candidate", type=float, default=None,
                        help="kill any single candidate's training after N seconds (recommended)")
    p_auto.add_argument("--hypothesis-seed", default="autoreason refinement loop",
                        help="initial hypothesis context shown to A baseline")
    p_auto.add_argument("--dry-run", action="store_true",
                        help="skip subprocess training; record noop outcomes")
    p_auto.add_argument("--llm-cli", choices=["hermes", "claude", "codex"], default=None,
                        help="which local LLM CLI to invoke (default: AUTORESEARCH_LLM_CLI env or 'hermes')")
    p_auto.add_argument("--llm-model", default=None,
                        help="model name passed to the selected CLI (e.g. 'anthropic/claude-sonnet-4' for hermes)")
    p_auto.add_argument("--llm-provider", default="auto",
                        help="provider routing for hermes (ignored by claude/codex); default 'auto'")

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
    if args.cmd == "autoreason":
        return cmd_autoreason(
            args.target,
            max_passes=args.max_passes,
            convergence=args.convergence,
            max_seconds_per_candidate=args.max_seconds_per_candidate,
            hypothesis_seed=args.hypothesis_seed,
            dry_run=args.dry_run,
            llm_cli=args.llm_cli,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
