"""AuthorBAgent — produces a unified-diff patch addressing the critic's findings.

Per autoreason paper §2: the author sees A and the critique only — no
drafting history. Output is a unified diff that `git apply --check` accepts.
The system prompt explicitly forbids V1 file edits (defense in depth — the
patch applicator also rejects them, but baking the rule into the prompt
reduces wasted LLM calls and clearer audit trails).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_loop.agents.client import LLMClient
from research_loop.targets._base import V1_FORBIDDEN_PATHS


_FORBIDDEN_LIST = "\n".join(f"  - {p}" for p in V1_FORBIDDEN_PATHS)


AUTHOR_B_SYSTEM_PROMPT = f"""You are a senior ML engineer revising a training script based on a structured critique.

Your job: produce a unified diff that addresses the critique's identified problems. You will see the current trainer source (A) and a critique with concrete problems. You write the revision (B).

Rules:

1. Address each problem the critique identifies. If the critique says "No actionable problems", produce an empty diff (just the words "NO_PATCH" on its own line — see Output format below).

2. Do not make changes outside the identified problems. No drive-by refactors, comment fixes, or "while I'm here" cleanup. Surgical changes only.

3. Produce a SINGLE unified diff applicable via `git apply` from the repo root. Path style: `a/path/to/file.py` and `b/path/to/file.py`. Include `diff --git a/... b/...` headers and full hunk context (3 lines minimum).

4. **NEVER edit any of these V1-protected files:**
{_FORBIDDEN_LIST}

   These files preserve V1 experiment history. Edits to them will be rejected before training runs and waste your effort. If the critique points at a problem in one of these files, your patch should modify the V2 trainer (train_v2.py / train_dino_v2.py) to work around the V1 issue, not fix it directly in V1.

5. Prefer minimal changes. A 5-line patch that addresses one root cause beats a 50-line refactor. The smaller the diff, the easier to revert if the experiment fails.

6. Don't change the trainer's CLI or top-level constants unless the critique specifically points at them. The tournament's job is to A/B/AB-test changes; sweeping rewrites defeat that.

Output format. Return EXACTLY this structure:

# Rationale
One short paragraph (2-4 sentences) explaining what you changed and why, tied to specific items in the critique.

# Diff
```
<unified diff goes here>
```

If the critique reports no actionable problems and you should not patch:

# Rationale
No changes proposed — incumbent is performing within expectations.

# Diff
NO_PATCH
"""


@dataclass(frozen=True)
class PatchProposal:
    rationale: str
    diff: str  # unified diff text; empty string when NO_PATCH
    raw: str   # full LLM response for audit


_FENCE_RE = re.compile(r"```(?:diff)?\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _parse_patch_proposal(raw: str) -> PatchProposal:
    """Pull `# Rationale` paragraph and the diff block out of the response."""
    rationale = ""
    diff = ""
    section: str | None = None
    rationale_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Rationale"):
            section = "rationale"
            continue
        if stripped.startswith("# Diff"):
            section = "diff"
            continue
        if section == "rationale" and stripped:
            rationale_lines.append(stripped)
    rationale = " ".join(rationale_lines)

    # Diff: prefer fenced block; fall back to "NO_PATCH" sentinel
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        diff_body = fence_match.group("body").strip()
        diff = "" if diff_body.upper() == "NO_PATCH" else (diff_body + "\n")
    else:
        # Look for unfenced "NO_PATCH" sentinel after # Diff heading
        if "NO_PATCH" in raw.split("# Diff", 1)[-1]:
            diff = ""
        else:
            # No fence and no NO_PATCH — best-effort: take everything after `# Diff`
            after = raw.split("# Diff", 1)[-1].strip()
            stripped = after.strip("`").strip()
            diff = (stripped + "\n") if stripped else ""

    return PatchProposal(rationale=rationale, diff=diff, raw=raw)


class AuthorBAgent:
    """Critic output + trainer source → unified-diff patch B."""

    SYSTEM_PROMPT = AUTHOR_B_SYSTEM_PROMPT

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def author(
        self,
        *,
        trainer_source: str,
        trainer_path: str,
        critique_text: str,
    ) -> PatchProposal:
        """Single fresh-agent call. Returns parsed PatchProposal with raw text preserved.

        `trainer_path` is the repo-relative path Author B should target in
        the diff headers (e.g. "student_finetune/train_v2.py").
        """
        user = (
            f"## Current trainer source ({trainer_path})\n"
            f"```python\n{trainer_source}\n```\n\n"
            "## Critique\n"
            f"{critique_text}\n\n"
            f"Produce the patch in the format specified by your system prompt. "
            f"Diff paths must be `a/{trainer_path}` and `b/{trainer_path}`."
        )
        # Higher temperature on the author — encourage diverse revisions
        # (autoreason paper uses 0.8 for author roles).
        raw = self.client.call(self.SYSTEM_PROMPT, user, temperature=0.8)
        return _parse_patch_proposal(raw)
