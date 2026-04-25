"""CriticAgent — surfaces problems in the incumbent training script.

Per autoreason paper §2: critics find problems only, no fixes proposed.
This separation is load-bearing — when the same agent both criticizes and
fixes, it tends to invent flaws to satisfy the critique prompt (sycophancy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_loop.agents.client import AgentClient


CRITIC_SYSTEM_PROMPT = """You are a senior ML engineer reviewing a production retail-product re-identification training script and its recent experiment history. Your only job is to identify concrete problems — do not propose fixes.

Your reviewing posture:
  - Be specific. Reference exact symbols, line ranges, or metric values from the inputs.
  - Be evidence-based. Tie each problem to something visible in the trainer source, the recent results_v2.tsv rows, or the recent Outcome metrics.
  - Be honest about ambiguity. If a metric is plateauing but the code looks reasonable, say "plateau on metric X with no obvious code-side cause" rather than inventing a flaw.
  - If the run is healthy, say so explicitly. "No actionable problems" is a valid critique.

You will be given:
  - The current trainer source file
  - The last 30 rows of results_v2.tsv (round outcomes)
  - The last 10 Outcome JSON records from history.jsonl

Output format. Return a critique in this exact shape (Markdown headings preserved):

# Summary
One paragraph (2-4 sentences) describing the overall state — is the run healthy, plateauing, regressing, or showing some specific failure mode?

# Problems
- Problem 1: <concrete issue, with reference>
- Problem 2: ...
- ...

If there are no actionable problems, write a single line under # Problems: "No actionable problems — incumbent is performing within expectations."

Do not propose solutions. Do not include code. Do not speculate beyond what the inputs support."""


@dataclass(frozen=True)
class Critique:
    summary: str
    problems: list[str]
    raw: str  # full LLM response for audit


def _parse_critique(raw: str) -> Critique:
    """Extract # Summary and # Problems sections from the LLM response."""
    summary = ""
    problems: list[str] = []
    section: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Summary"):
            section = "summary"
            continue
        if stripped.startswith("# Problems"):
            section = "problems"
            continue
        if not stripped:
            continue
        if section == "summary":
            summary = (summary + " " + stripped).strip() if summary else stripped
        elif section == "problems":
            # Bulleted line — strip leading marker
            if stripped.startswith(("- ", "* ", "• ")):
                problems.append(stripped[2:].strip())
            elif stripped[0:2].rstrip(".").isdigit():
                problems.append(stripped.split(".", 1)[-1].strip())
    return Critique(summary=summary, problems=problems, raw=raw)


class CriticAgent:
    """Reads trainer + history → surfaces concrete problems."""

    SYSTEM_PROMPT = CRITIC_SYSTEM_PROMPT

    def __init__(self, client: AgentClient) -> None:
        self.client = client

    def critique(
        self,
        *,
        trainer_source: str,
        results_tsv_tail: str,
        recent_outcomes_json: str,
    ) -> Critique:
        """Single fresh-agent call. Returns parsed Critique with raw text preserved."""
        user = (
            "## Trainer source\n"
            f"```python\n{trainer_source}\n```\n\n"
            "## Recent results_v2.tsv (last 30 rows)\n"
            f"```tsv\n{results_tsv_tail}\n```\n\n"
            "## Recent Outcome records (last 10)\n"
            f"```jsonl\n{recent_outcomes_json}\n```\n\n"
            "Produce the critique in the format specified by your system prompt."
        )
        # Lower temperature for the critic — we want consistent problem-finding,
        # not creative interpretation (autoreason paper §2 uses 0.3 for judges).
        raw = self.client.call(self.SYSTEM_PROMPT, user, temperature=0.3)
        return _parse_critique(raw)
