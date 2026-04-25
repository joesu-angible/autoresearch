"""SynthesizerAgent — produces AB by taking the strongest elements of A and B.

Per autoreason paper §2: the synthesizer sees A and B with anonymized labels
and no metric history. The point is to find a more conservative middle ground
than B alone — half the change, restricted to the strongest part — without
being biased by either side's prior performance numbers.

Per project decision 2026-04-25 (SPEC §Open questions resolved): the
synthesizer sees ONLY the patches, not metric history.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_loop.agents.client import AgentClient
from research_loop.agents.author import (
    PatchProposal,
    _FENCE_RE,
)
from research_loop.targets._base import V1_FORBIDDEN_PATHS


_FORBIDDEN_LIST = "\n".join(f"  - {p}" for p in V1_FORBIDDEN_PATHS)


SYNTHESIZER_SYSTEM_PROMPT = f"""You are a senior ML engineer producing a conservative synthesis of two competing revisions to a training script.

You will see two patches labeled X and Y (anonymized — you do not know which one was authored by the critic-driven revision and which is the do-nothing baseline). Your job is to produce a third patch (the synthesis) that takes the strongest elements of each but commits less aggressively than the more extensive of the two.

Synthesis principles:

1. Find the smallest subset of changes that preserves the most likely-correct mechanism. If X adds three changes and Y adds none, your synthesis might keep one of X's changes — the one with the clearest rationale — and drop the others.

2. Halve magnitudes when possible. If X changes a weight from 0.02 to 0.10, your synthesis might propose 0.05.

3. Prefer additions over removals. If X removes a feature, your synthesis is more likely to keep the feature but reduce its weight than to remove it.

4. Be willing to produce an empty patch ("NO_PATCH") when X and Y are both empty, or when X's changes are too entangled to halve cleanly. The tournament treats no-op as a first-class option.

5. **NEVER edit any of these V1-protected files:**
{_FORBIDDEN_LIST}

   Defense in depth — the patch applicator will reject V1 edits, but you should not produce them.

6. Output a SINGLE unified diff applicable via `git apply` from the repo root. Path style: `a/path/to/file.py` and `b/path/to/file.py`.

You do NOT see the patches' performance metrics. Synthesize on structure and risk, not on which one looked better in past runs.

Output format. Return EXACTLY this structure:

# Rationale
One short paragraph (2-4 sentences) explaining what you kept from each side and what you halved or dropped.

# Diff
```
<unified diff goes here>
```

If the synthesis would be empty:

# Rationale
Synthesis is empty — neither candidate proposes changes that combine cleanly at lower magnitude.

# Diff
NO_PATCH
"""


def _parse_synthesis(raw: str) -> PatchProposal:
    """Same shape as AuthorBAgent's PatchProposal — uses the same parser."""
    rationale = ""
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

    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        diff_body = fence_match.group("body").strip()
        diff = "" if diff_body.upper() == "NO_PATCH" else diff_body
    else:
        if "NO_PATCH" in raw.split("# Diff", 1)[-1]:
            diff = ""
        else:
            after = raw.split("# Diff", 1)[-1].strip()
            diff = after.strip("`").strip()

    return PatchProposal(rationale=rationale, diff=diff, raw=raw)


class SynthesizerAgent:
    """Two patches with anonymized labels → conservative synthesis patch (AB)."""

    SYSTEM_PROMPT = SYNTHESIZER_SYSTEM_PROMPT

    def __init__(self, client: AgentClient) -> None:
        self.client = client

    def synthesize(
        self,
        *,
        patch_x: str,
        patch_y: str,
        trainer_path: str,
    ) -> PatchProposal:
        """Single fresh-agent call. Anonymized X / Y labels per autoreason §2."""
        user = (
            "## Patch X\n"
            f"```\n{patch_x or 'NO_PATCH'}\n```\n\n"
            "## Patch Y\n"
            f"```\n{patch_y or 'NO_PATCH'}\n```\n\n"
            f"Produce the synthesis patch in the format specified by your system prompt. "
            f"Diff paths must be `a/{trainer_path}` and `b/{trainer_path}`."
        )
        raw = self.client.call(self.SYSTEM_PROMPT, user, temperature=0.8)
        return _parse_synthesis(raw)
