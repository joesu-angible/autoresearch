"""Single-shared Anthropic SDK wrapper for autoreason agent calls.

Each `.call(system, user)` is a fresh single-shot Messages request — no chat
history persisted across calls — so the autoreason "fresh agent" invariant
holds even though all three roles share one client instance. The system block
gets `cache_control: ephemeral` so the prompt cache amortizes the role's
fixed instructions across passes within a 5-minute window.

API key sources (first match wins):
  1. ANTHROPIC_API_KEY environment variable
  2. ~/.hermes/.env file (key = value lines), to align with the repo's
     existing Hermes Agent integration

Default model is `claude-sonnet-4-6` per SPEC §Tech stack — Sonnet 4.6
is fast enough for hour-scale autoreason loops and capable enough on ML
training code. Override per-instance via the `model` constructor arg.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


def _load_dotenv(path: Path) -> dict[str, str]:
    """Tolerant .env parser — KEY=VALUE per line, # comments, no quotes magic."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def resolve_api_key() -> str:
    """Find an Anthropic API key in env or in ~/.hermes/.env. Raise if missing."""
    if (key := os.environ.get("ANTHROPIC_API_KEY")):
        return key
    hermes = _load_dotenv(Path.home() / ".hermes" / ".env")
    if (key := hermes.get("ANTHROPIC_API_KEY")):
        return key
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Set the environment variable or "
        "add it to ~/.hermes/.env. autoreason cannot run without LLM access."
    )


class AgentClient:
    """Thin wrapper around `anthropic.Anthropic` for the three autoreason roles.

    Each `.call()` is a fresh single-shot — no chat history. System prompts
    are sent with `cache_control: ephemeral` so repeated calls with the
    same role-specific system text hit the cache.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or resolve_api_key())

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> str:
        """Run one fresh-agent inference; return the raw assistant text.

        Per autoreason paper §2: temperature 0.8 for authors / synthesizer
        (encourage diverse revisions); 0.3 for judges (consistent evaluation).
        Caller picks the right value for its role.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return next(block.text for block in response.content if block.type == "text")
