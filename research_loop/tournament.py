"""Autoreason tournament CLI — propose / rank / run / promote.

Subcommands:
  propose --target {student_v2,dino_v2}
      Emits a starter A/B/AB triple to research_loop/history.jsonl. The B and AB
      candidates are template stubs the user fills in (hypothesis, patch, etc.).

  rank
      Reads outstanding candidates and prints judge ranking.

  run --candidate <id>
      Runs the named candidate end-to-end via its target adapter. For A
      (do-nothing) this is a no-op that records the incumbent's result.

  promote --round <round_id>
      Apply guardrails over a round's results and decide the winner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_loop.candidate import (
    Candidate,
    append_history,
    read_history,
)
from research_loop.judges import rank as rank_candidates
from research_loop.targets import DinoV2Target, StudentV2Target

HISTORY_PATH = Path(__file__).resolve().parent / "history.jsonl"

TARGETS = {
    "student_v2": StudentV2Target,
    "dino_v2": DinoV2Target,
}


def cmd_propose(target_name: str, hypothesis: str = "") -> int:
    """Emit a starter A/B/AB triple. B/AB are placeholders to be filled in."""
    parent_id = "incumbent-baseline"
    a = Candidate(
        kind="A",
        target=target_name,
        hypothesis="do nothing — keep the current best of results_v2.tsv",
        expected_metric="combined unchanged",
        changed_files=[],
        risks=[],
        rollback="N/A — incumbent baseline",
        patch="",
    )
    b = Candidate(
        kind="B",
        target=target_name,
        hypothesis=hypothesis or "TODO: fill in concrete hypothesis",
        expected_metric="combined +0.005 (TODO: replace)",
        changed_files=["student_finetune/train_v2.py" if target_name == "student_v2" else "dino_finetune/train_dino_v2.py"],
        risks=["may regress recall@1"],
        rollback="combined < incumbent - 0.005",
        patch="--- a/placeholder\n+++ b/placeholder\n@@\n+# TODO: real patch\n",
        parent_incumbent_id=parent_id,
    )
    ab = Candidate(
        kind="AB",
        target=target_name,
        hypothesis="conservative synthesis — apply B at half strength",
        expected_metric="combined +0.002 (TODO)",
        changed_files=b.changed_files,
        risks=["may underdeliver"],
        rollback=b.rollback,
        patch="--- a/placeholder\n+++ b/placeholder\n@@\n+# TODO: synthesis patch\n",
        parent_incumbent_id=parent_id,
    )
    for c in (a, b, ab):
        append_history(HISTORY_PATH, c)
    print(f"Proposed 3 candidates for target={target_name}:")
    for c in (a, b, ab):
        print(f"  {c.kind} id={c.id}  hypothesis={c.hypothesis[:60]}")
    return 0


def cmd_rank() -> int:
    candidates = list(read_history(HISTORY_PATH))
    if not candidates:
        print("No candidates in history.")
        return 0
    ordering = rank_candidates(candidates)
    print(f"Ranking ({len(ordering)} candidates):")
    for o in ordering:
        print(f"  score={o.score:.3f}  kind={o.kind}  id={o.candidate_id}")
    return 0


def cmd_run(candidate_id: str, dry_run: bool = True) -> int:
    candidates = list(read_history(HISTORY_PATH))
    match = next((c for c in candidates if c.id == candidate_id), None)
    if match is None:
        print(f"Candidate not found: {candidate_id}", file=sys.stderr)
        return 2
    target_cls = TARGETS.get(match.target)
    if target_cls is None:
        print(f"Unknown target: {match.target}", file=sys.stderr)
        return 2
    adapter = target_cls()
    adapter.apply_patch(match)
    if match.kind == "A":
        print(f"[A=do-nothing] no GPU work for candidate {match.id}")
        return 0
    rc, log = adapter.train(max_epochs=1, dry_run=dry_run)
    print(log)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_loop.tournament")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_propose = sub.add_parser("propose")
    p_propose.add_argument("--target", required=True, choices=list(TARGETS))
    p_propose.add_argument("--hypothesis", default="")
    sub.add_parser("rank")
    p_run = sub.add_parser("run")
    p_run.add_argument("--candidate", required=True)
    p_run.add_argument("--dry-run", action="store_true", default=True)
    p_run.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = p.parse_args(argv)

    if args.cmd == "propose":
        return cmd_propose(args.target, args.hypothesis)
    if args.cmd == "rank":
        return cmd_rank()
    if args.cmd == "run":
        return cmd_run(args.candidate, dry_run=args.dry_run)
    return 2


if __name__ == "__main__":
    sys.exit(main())
