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
import os
import subprocess
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
from research_loop.variants import VariantSpec, load_variants

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


def _make_variant_candidate(target: str, round_id: str, parent_id: str, spec: VariantSpec) -> Candidate:
    """Materialize one operator-supplied variant into a kind='B' Candidate.

    Each variant in a sweep round is kind='B'; uniqueness is by id, not kind.
    """
    return Candidate(
        kind="B",
        target=target,
        round_id=round_id,
        hypothesis=spec["hypothesis"],
        expected_metric=spec["expected_metric"],
        changed_files=list(spec["changed_files"]),
        risks=list(spec["risks"]),
        rollback=spec["rollback"],
        patch=spec["patch"],
        parent_incumbent_id=parent_id,
    )


def cmd_propose(target: str, hypothesis: str, baseline_only: bool,
                round_id: str | None = None, variants: Path | None = None) -> int:
    if target not in TARGETS:
        print(f"Unknown target: {target}", file=sys.stderr)
        return 2
    if variants is not None and baseline_only:
        # argparse's mutually_exclusive_group catches the CLI case; this guards
        # programmatic callers.
        print("--variants and --baseline-only are mutually exclusive", file=sys.stderr)
        return 2
    rid = round_id or new_round_id()
    a = _make_baseline_a(target, rid, hypothesis)
    append_history(HISTORY_PATH, a)
    print(f"Round {rid} proposed for target={target}:")
    print(f"  A  id={a.id}  {a.hypothesis[:60]}")
    if baseline_only:
        return 0
    if variants is not None:
        specs = load_variants(variants)
        if not specs:
            print(f"  warning: variants file {variants} contained no entries", file=sys.stderr)
            return 0
        for spec in specs:
            cand = _make_variant_candidate(target, rid, parent_id=a.id, spec=spec)
            append_history(HISTORY_PATH, cand)
            print(f"  B  id={cand.id}  {cand.hypothesis[:60]}")
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
                  baseline_only: bool, dry_run: bool,
                  variants: Path | None = None) -> int:
    rid = new_round_id()
    rc = cmd_propose(target, hypothesis,
                     baseline_only=baseline_only, round_id=rid, variants=variants)
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
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _new_run_id() -> str:
    """Sortable run id for the run directory."""
    import time
    import uuid
    return f"run{int(time.time())}-{uuid.uuid4().hex[:6]}"


class _Tee:
    """Mirror writes to two streams. Used to send autoreason's narrative print()
    output to both stdout and run_dir/autoreason.log so external bots (Hermes,
    Slack) can tail one canonical log path discovered via summary.json.
    """

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, s):
        self.primary.write(s)
        self.primary.flush()
        self.secondary.write(s)
        self.secondary.flush()
        return len(s)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self):  # subprocess sometimes asks
        return self.primary.fileno()


def _write_run_summary(
    run_dir: Path,
    *,
    run_id: str,
    target: str,
    started_at: str,
    pass_index: int,
    max_passes: int,
    consecutive_a_wins: int,
    convergence: int,
    last_decision: dict | None,
    best_so_far: dict | None,
    latest_critique_summary: str,
    status: str,
) -> None:
    """Atomic JSON dump consumed by `tournament status` and external bots.

    Atomic via temp + rename so a concurrent reader never sees a half-written file.
    """
    import json
    payload = {
        "run_id": run_id,
        "target": target,
        "started_at": started_at,
        "current_pass": pass_index,
        "max_passes": max_passes,
        "consecutive_a_wins": consecutive_a_wins,
        "convergence_threshold": convergence,
        "last_decision": last_decision,
        "best_so_far": best_so_far,
        "latest_critique_summary": latest_critique_summary,
        "status": status,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.json"
    tmp = summary.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(summary)


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
    # Global defaults (inherited by all three roles unless overridden below)
    llm_cli: str | None = None,
    llm_model: str | None = None,
    llm_provider: str = "auto",
    # Per-role overrides — None means inherit the global default. Per
    # autoreason paper §7.3, mixing models across roles is well-supported
    # (e.g. cheap author + strong judge); we expose the same flexibility.
    critic_cli: str | None = None,
    critic_model: str | None = None,
    author_cli: str | None = None,
    author_model: str | None = None,
    synthesizer_cli: str | None = None,
    synthesizer_model: str | None = None,
    agent_client_factory=None,  # test injection; takes (role_name) → LLMClient
) -> int:
    """Fully autonomous autoreason loop.

    Each pass: fresh Critic → fresh Author B → fresh Synthesizer → run A/B/AB
    via the target adapter (under per-candidate time budget) → promote.decide().
    Winner becomes the new A; loop terminates after `convergence` consecutive
    A wins or after `max_passes` total passes.

    LLM access is via subprocess to a local CLI (hermes / claude / codex).
    Each role can use a different CLI and/or model — autoreason paper §7.3
    showed mixed-model setups (cheap author + strong judge) are first-class.
    Per-role values default to the global llm_cli / llm_model / llm_provider.

    `agent_client_factory` is injected for testing — production callers pass
    None and we construct a real CLI-backed client per role.
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

    # Resolve per-role config (fall back to globals)
    role_config = {
        "critic":       (critic_cli       or llm_cli, critic_model       or llm_model),
        "author":       (author_cli       or llm_cli, author_model       or llm_model),
        "synthesizer":  (synthesizer_cli  or llm_cli, synthesizer_model  or llm_model),
    }

    if agent_client_factory is None:
        from research_loop.agents.client import make_llm_client
        def _factory(role: str):
            cli, model = role_config[role]
            return make_llm_client(cli, model=model, provider=llm_provider)
        agent_client_factory = _factory

    critic_client = agent_client_factory("critic")
    author_client = agent_client_factory("author")
    synthesizer_client = agent_client_factory("synthesizer")
    for role, c in (("critic", critic_client), ("author", author_client),
                    ("synthesizer", synthesizer_client)):
        cli, model = role_config[role]
        print(f"  {role:12s} → {getattr(c, 'name', type(c).__name__)}"
              + (f" model={model}" if model else ""))

    from research_loop.agents.author import AuthorBAgent
    from research_loop.agents.critic import CriticAgent
    from research_loop.agents.synthesizer import SynthesizerAgent

    critic = CriticAgent(critic_client)
    author = AuthorBAgent(author_client)
    synthesizer = SynthesizerAgent(synthesizer_client)

    # Run-level state for the status surface (Slice 1 of T9)
    import datetime
    run_id = _new_run_id()
    run_dir = RUNS_DIR / run_id
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    current_pointer = RUNS_DIR / f"{target}_CURRENT.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    current_pointer.write_text(run_id)

    # Mirror cmd_autoreason's narrative output to run_dir/autoreason.log so
    # external tooling (Hermes, Slack bots, the `tournament status` command)
    # can tail one canonical path discovered via summary.json. Operators who
    # also nohup-redirect get duplicate output — that's fine.
    # Write PID file for `tournament status` to detect alive vs exited
    (run_dir / "autoreason.pid").write_text(str(os.getpid()))

    narrative_log_path = run_dir / "autoreason.log"
    _log_fh = open(narrative_log_path, "a", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_fh)
    sys.stderr = _Tee(_orig_stderr, _log_fh)

    def _close_narrative_log() -> None:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        _log_fh.close()

    def _status_dump(
        pass_index: int, status: str, latest_critique: str = "",
        last_decision_dict: dict | None = None, best: dict | None = None,
    ) -> None:
        _write_run_summary(
            run_dir, run_id=run_id, target=target, started_at=started_at,
            pass_index=pass_index, max_passes=max_passes,
            consecutive_a_wins=consecutive_a_wins, convergence=convergence,
            last_decision=last_decision_dict, best_so_far=best,
            latest_critique_summary=latest_critique, status=status,
        )

    consecutive_a_wins = 0
    last_round_id: str | None = None
    latest_critique_text = ""
    last_decision_dict: dict | None = None
    best_so_far: dict | None = None

    _status_dump(0, "starting")
    print(f"  run_id: {run_id}")
    print(f"  status: {run_dir}/summary.json")

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
            _status_dump(pass_index, "promote_failed", latest_critique_text, last_decision_dict, best_so_far)
            _close_narrative_log()
            return rc

        # 7. Convergence check
        decisions = [d for d in __import__("research_loop.candidate", fromlist=["read_decisions"]).read_decisions(HISTORY_PATH, round_id=round_id)]
        if not decisions:
            print(f"  no decision recorded for round {round_id}", file=sys.stderr)
            _status_dump(pass_index, "no_decision", latest_critique_text, last_decision_dict, best_so_far)
            _close_narrative_log()
            return 1
        decision = decisions[-1]
        if decision.winner_kind == "A":
            consecutive_a_wins += 1
        else:
            consecutive_a_wins = 0
        last_round_id = round_id
        latest_critique_text = critique.summary
        last_decision_dict = {
            "round_id": decision.round_id,
            "winner_id": decision.winner_id,
            "winner_kind": decision.winner_kind,
            "promote": decision.promote,
            "deployable": decision.deployable,
            "reason": decision.reason,
        }
        # Best so far: pull from the freshest A or B/AB outcome we've seen
        from research_loop.candidate import read_outcomes as _ro
        all_outcomes = list(_ro(HISTORY_PATH))
        best_so_far = None
        for o in reversed(all_outcomes):
            if o.target != target or o.status != "success" or not o.metrics:
                continue
            if best_so_far is None or o.metrics.get("combined", 0) > best_so_far.get("combined", 0):
                best_so_far = dict(o.metrics)
        _status_dump(pass_index, "running", latest_critique_text, last_decision_dict, best_so_far)
        print(f"  decision: winner={decision.winner_kind}, consecutive A={consecutive_a_wins}/{convergence}")

        if consecutive_a_wins >= convergence:
            print(f"\nConverged at pass {pass_index} (A won {convergence} consecutive rounds).")
            _status_dump(pass_index, "converged", latest_critique_text, last_decision_dict, best_so_far)
            _close_narrative_log()
            return 0

    print(f"\nMax passes ({max_passes}) reached without convergence. Last round: {last_round_id}")
    _status_dump(max_passes, "max_passes_exhausted", latest_critique_text, last_decision_dict, best_so_far)
    _close_narrative_log()
    return 1


# ---------------------------------------------------------------------------
# status — one-shot human/bot-readable summary of an autoreason run
# ---------------------------------------------------------------------------

def _format_status(summary: dict, *, run_dir: Path, log_path: Path,
                   pid_alive: bool | None, etime: str | None,
                   history_count: int) -> str:
    """Render summary.json + ambient state into a single readable block.
    Designed to be `cat`-ed by an external bot (Hermes, Slack) and pasted
    verbatim back to a user asking "how is training going?".
    """
    lines: list[str] = []
    target = summary.get("target", "?")
    lines.append(f"autoreason status — {target}")
    pid_str = "unknown"
    if pid_alive is True:
        pid_str = f"alive ({etime or '?'})"
    elif pid_alive is False:
        pid_str = "exited"
    lines.append(f"  Run:          {summary.get('run_id', '?')} ({pid_str})")
    lines.append(f"  Started:      {summary.get('started_at', '?')}")
    lines.append(f"  Status:       {summary.get('status', '?')}")
    pass_idx = summary.get("current_pass", 0)
    max_p = summary.get("max_passes", "?")
    a_wins = summary.get("consecutive_a_wins", 0)
    conv = summary.get("convergence_threshold", "?")
    lines.append(f"  Pass:         {pass_idx} / {max_p}  (consecutive A wins: {a_wins} / {conv})")
    last_dec = summary.get("last_decision")
    if last_dec:
        lines.append(
            f"  Last decision: {last_dec.get('winner_kind', '?')} wins "
            f"({last_dec.get('round_id', '?')}, deployable={last_dec.get('deployable', '?')})"
        )
        lines.append(f"                 reason: {last_dec.get('reason', '')[:120]}")
    best = summary.get("best_so_far")
    if best:
        lines.append(
            f"  Best so far:  combined={best.get('combined', 0):.4f}  "
            f"recall@1={best.get('recall_1', 0):.4f}  "
            f"neg_acc={best.get('productness_neg_acc', 0):.4f}"
        )
    crit = summary.get("latest_critique_summary", "")
    if crit:
        lines.append(f"  Latest critique: {crit[:200]}")
    lines.append("  Logs:")
    lines.append(f"    narrative: {log_path}")
    lines.append(f"    history:   {HISTORY_PATH} ({history_count} records)")
    lines.append(f"    run dir:   {run_dir}")
    return "\n".join(lines)


def cmd_status(target: str | None = None, run_id: str | None = None) -> int:
    """Print a one-shot status block — readable by humans and external bots.

    Resolution order for which run to summarize:
      1. explicit --run RUN_ID
      2. --target TARGET → CURRENT pointer for that target
      3. most recently mtime'd run dir under research_loop/runs/

    Exit codes:
      0  found a run, printed status
      2  no run found
    """
    import json
    if not RUNS_DIR.exists():
        print(f"No runs directory at {RUNS_DIR}", file=sys.stderr)
        return 2

    # Resolve run_dir
    chosen_run_dir: Path | None = None
    if run_id:
        candidate = RUNS_DIR / run_id
        if candidate.is_dir():
            chosen_run_dir = candidate
    elif target:
        pointer = RUNS_DIR / f"{target}_CURRENT.txt"
        if pointer.exists():
            chosen_run_dir = RUNS_DIR / pointer.read_text().strip()
    else:
        # Newest mtime wins
        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
        if run_dirs:
            chosen_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)

    if chosen_run_dir is None or not chosen_run_dir.is_dir():
        print(f"No autoreason run found (target={target}, run_id={run_id})", file=sys.stderr)
        return 2

    summary_path = chosen_run_dir / "summary.json"
    if not summary_path.exists():
        print(f"Run dir {chosen_run_dir} has no summary.json yet", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text())

    # Best-effort process check — if a PID was recorded, see if it's alive.
    pid_alive: bool | None = None
    etime: str | None = None
    pid_path = chosen_run_dir / "autoreason.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            import os
            os.kill(pid, 0)
            pid_alive = True
            try:
                ps = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "etime="],
                    capture_output=True, text=True, check=False,
                )
                etime = ps.stdout.strip() or None
            except Exception:
                pass
        except (ProcessLookupError, ValueError, PermissionError):
            pid_alive = False

    history_count = 0
    if HISTORY_PATH.exists():
        history_count = sum(1 for _ in HISTORY_PATH.open() if _.strip())

    print(_format_status(
        summary,
        run_dir=chosen_run_dir,
        log_path=chosen_run_dir / "autoreason.log",
        pid_alive=pid_alive,
        etime=etime,
        history_count=history_count,
    ))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_loop.tournament")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_propose = sub.add_parser("propose", help="emit A/B/AB candidates for a new round")
    p_propose.add_argument("--target", required=True, choices=list(TARGETS))
    p_propose.add_argument("--hypothesis", default="")
    propose_mode = p_propose.add_mutually_exclusive_group()
    propose_mode.add_argument("--baseline-only", action="store_true",
                              help="emit only A (incumbent); skip B/AB placeholders")
    propose_mode.add_argument("--variants", type=Path, default=None,
                              help="JSONL file: emit A + one Candidate per line (sweep mode)")

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
    round_mode = p_round.add_mutually_exclusive_group()
    round_mode.add_argument("--baseline-only", action="store_true",
                            help="run only the A (incumbent baseline); useful for first run")
    round_mode.add_argument("--variants", type=Path, default=None,
                            help="JSONL file: run A + one Candidate per line (sweep mode)")
    p_round.add_argument("--dry-run", action="store_true")

    p_status = sub.add_parser("status",
        help="one-shot 'how is autoreason going?' summary (for humans + Slack bots)")
    p_status.add_argument("--target", choices=["student_v2", "dino_v2"], default=None,
                          help="resolve via runs/<target>_CURRENT.txt")
    p_status.add_argument("--run", default=None,
                          help="explicit run_id (overrides --target)")

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
                        help="default CLI for all 3 roles (env: AUTORESEARCH_LLM_CLI; default 'hermes')")
    p_auto.add_argument("--llm-model", default=None,
                        help="default model for all 3 roles (e.g. 'anthropic/claude-sonnet-4' for hermes)")
    p_auto.add_argument("--llm-provider", default="auto",
                        help="provider routing for hermes (ignored by claude/codex); default 'auto'")
    # Per-role overrides — autoreason paper §7.3 supports mixed-model setups
    # (cheap author + strong judge). Each defaults to the global --llm-cli/--llm-model.
    p_auto.add_argument("--critic-cli", choices=["hermes", "claude", "codex"], default=None,
                        help="override CLI for the Critic role (analytical; benefits from a strong model)")
    p_auto.add_argument("--critic-model", default=None,
                        help="override model for the Critic role")
    p_auto.add_argument("--author-cli", choices=["hermes", "claude", "codex"], default=None,
                        help="override CLI for the Author B role (creative patch generation)")
    p_auto.add_argument("--author-model", default=None,
                        help="override model for the Author B role")
    p_auto.add_argument("--synthesizer-cli", choices=["hermes", "claude", "codex"], default=None,
                        help="override CLI for the Synthesizer role (conservative AB)")
    p_auto.add_argument("--synthesizer-model", default=None,
                        help="override model for the Synthesizer role")

    args = p.parse_args(argv)

    if args.cmd == "propose":
        return cmd_propose(args.target, args.hypothesis,
                           baseline_only=args.baseline_only,
                           variants=args.variants)
    if args.cmd == "rank":
        return cmd_rank(args.round)
    if args.cmd == "run":
        return cmd_run(args.candidate, args.epochs, args.dry_run)
    if args.cmd == "promote":
        return cmd_promote(args.round)
    if args.cmd == "run-round":
        return cmd_run_round(args.target, args.hypothesis, args.epochs,
                             baseline_only=args.baseline_only, dry_run=args.dry_run,
                             variants=args.variants)
    if args.cmd == "status":
        return cmd_status(target=args.target, run_id=args.run)
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
            critic_cli=args.critic_cli,
            critic_model=args.critic_model,
            author_cli=args.author_cli,
            author_model=args.author_model,
            synthesizer_cli=args.synthesizer_cli,
            synthesizer_model=args.synthesizer_model,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
