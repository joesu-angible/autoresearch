"""Manual real-LLM smoke for the autoreason loop (T8).

Runs ONE autoreason pass against a real local LLM CLI (hermes, claude, or
codex) in dry-run mode (no GPU subprocess). Confirms:

  1. The chosen CLI is callable and returns text
  2. CriticAgent produces a parseable Critique with at least a summary
  3. AuthorBAgent produces a unified diff that `git apply --check` accepts
     (or NO_PATCH if the critic found nothing actionable)
  4. SynthesizerAgent produces a unified diff that `git apply --check`
     accepts (or NO_PATCH)
  5. Working tree is unchanged at the end (smoke only `--check`s, never applies)

Cost: ~3 CLI invocations. Per-call latency depends on which CLI / model is
selected.

Usage:
  # hermes (default — uses repo's existing autoresearch CLI)
  .venv/bin/python -m research_loop.tools.autoreason_smoke

  # claude with explicit model
  .venv/bin/python -m research_loop.tools.autoreason_smoke --llm-cli claude --llm-model claude-sonnet-4-6

  # codex
  .venv/bin/python -m research_loop.tools.autoreason_smoke --llm-cli codex

Requires a clean working tree. Safe to re-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-LLM smoke for the autoreason loop")
    parser.add_argument("--llm-cli", choices=["hermes", "claude", "codex"], default=None,
                        help="local LLM CLI (default: AUTORESEARCH_LLM_CLI env or 'hermes')")
    parser.add_argument("--llm-model", default=None,
                        help="model name forwarded to the selected CLI")
    parser.add_argument("--llm-provider", default="auto",
                        help="hermes provider routing (ignored by claude/codex)")
    args = parser.parse_args()

    print(f"=== autoreason real-LLM smoke ===")
    print(f"Repo: {REPO_ROOT}")
    sys.path.insert(0, str(REPO_ROOT))

    from research_loop.agents.client import make_llm_client

    # 2. Working tree must be clean — autoreason refuses dirty trees
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    if res.stdout.strip():
        print(f"FAIL: working tree dirty:\n{res.stdout}", file=sys.stderr)
        return 1
    print("Working tree clean.")

    # 3. Build agents — pick the CLI requested by the operator
    client = make_llm_client(args.llm_cli, model=args.llm_model, provider=args.llm_provider)
    print(f"LLM client: {client.name}" + (f" model={args.llm_model}" if args.llm_model else ""))
    from research_loop.agents.author import AuthorBAgent
    from research_loop.agents.critic import CriticAgent
    from research_loop.agents.synthesizer import SynthesizerAgent
    critic = CriticAgent(client)
    author = AuthorBAgent(client)
    synth = SynthesizerAgent(client)

    # 4. Read inputs (current student_v2 trainer + recent history if any)
    trainer_path = "student_finetune/train_v2.py"
    trainer_src = (REPO_ROOT / trainer_path).read_text()
    results_tsv = REPO_ROOT / "student_finetune" / "results_v2.tsv"
    results_tail = (
        "\n".join(results_tsv.read_text().splitlines()[-30:])
        if results_tsv.exists() else ""
    )
    history = REPO_ROOT / "research_loop" / "history.jsonl"
    recent_outcomes = ""
    if history.exists():
        outcomes = [
            line for line in history.read_text().splitlines()
            if '"record_type": "outcome"' in line
        ][-10:]
        recent_outcomes = "\n".join(outcomes)

    # 5. Critic
    print("\n[1/3] Critic...")
    critique = critic.critique(
        trainer_source=trainer_src,
        results_tsv_tail=results_tail,
        recent_outcomes_json=recent_outcomes,
    )
    print(f"  summary ({len(critique.summary)} chars): {critique.summary[:200]}")
    print(f"  problems: {len(critique.problems)}")
    for p in critique.problems[:3]:
        print(f"    - {p[:120]}")
    assert critique.summary, "Critic returned empty summary"

    # 6. Author B
    print("\n[2/3] Author B...")
    proposal = author.author(
        trainer_source=trainer_src,
        trainer_path=trainer_path,
        critique_text=critique.raw,
    )
    print(f"  rationale ({len(proposal.rationale)} chars): {proposal.rationale[:200]}")
    print(f"  diff length: {len(proposal.diff)} chars")
    if proposal.diff:
        # Verify it would apply
        check = subprocess.run(
            ["git", "apply", "--check"],
            cwd=REPO_ROOT, input=proposal.diff,
            capture_output=True, text=True, check=False,
        )
        if check.returncode != 0:
            print(f"  FAIL: Author B's diff does not apply:\n{check.stderr}", file=sys.stderr)
            print(f"\nDiff content:\n{proposal.diff}", file=sys.stderr)
            return 1
        print("  diff applies cleanly via `git apply --check`")
    else:
        print("  (NO_PATCH — critic found no actionable problems)")

    # 7. Synthesizer
    print("\n[3/3] Synthesizer...")
    synthesis = synth.synthesize(
        patch_x="",  # A is do-nothing
        patch_y=proposal.diff,
        trainer_path=trainer_path,
    )
    print(f"  rationale ({len(synthesis.rationale)} chars): {synthesis.rationale[:200]}")
    print(f"  diff length: {len(synthesis.diff)} chars")
    if synthesis.diff:
        check = subprocess.run(
            ["git", "apply", "--check"],
            cwd=REPO_ROOT, input=synthesis.diff,
            capture_output=True, text=True, check=False,
        )
        if check.returncode != 0:
            print(f"  FAIL: Synthesizer diff does not apply:\n{check.stderr}", file=sys.stderr)
            print(f"\nDiff content:\n{synthesis.diff}", file=sys.stderr)
            return 1
        print("  diff applies cleanly via `git apply --check`")
    else:
        print("  (NO_PATCH — synthesis is empty)")

    # 8. Working tree should still be clean (we only ran --check, never applied)
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert not res.stdout.strip(), f"Working tree dirty after smoke: {res.stdout}"
    print("\nWorking tree still clean after smoke.")

    # 9. Print summary for HANDOFF.md evidence
    print("\n" + "=" * 60)
    print("SMOKE PASSED")
    print(f"  Critic: {len(critique.problems)} problems found")
    print(f"  Author B: {'NO_PATCH' if not proposal.diff else f'{len(proposal.diff)} char diff'}")
    print(f"  Synthesizer: {'NO_PATCH' if not synthesis.diff else f'{len(synthesis.diff)} char diff'}")
    print(f"  Both diffs (if non-empty) pass `git apply --check`")
    print(f"  Working tree clean throughout")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
