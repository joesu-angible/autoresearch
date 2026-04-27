"""LLM client abstraction with multi-CLI support.

Per project decision 2026-04-25, autoreason calls go through one of the
user's existing local CLIs (`hermes`, `claude`, `codex`) — not via raw API
keys. This keeps auth, rate-limit handling, model selection, and provider
routing in the user's already-configured tools instead of duplicating them
in this repo.

Each CLI client subprocess-shells its CLI, sending a combined system+user
prompt and returning stdout. All three roles (Critic / Author B /
Synthesizer) share one client instance — the autoreason "fresh agent"
invariant is preserved because each `.call()` is a one-shot subprocess
with no shared state.

Selection (CLI flag → env → default):
  --llm-cli {hermes|claude|codex}    explicit choice
  AUTORESEARCH_LLM_CLI=hermes        env override
  default: hermes (matches repo's existing autoresearch convention)

Per-CLI model override:
  --llm-model NAME                   passed through to whichever CLI
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

LlmCliName = Literal["hermes", "claude", "codex"]
DEFAULT_CLI: LlmCliName = "hermes"

# How long to wait for one CLI invocation. Generous because some CLIs do
# tool-calling internally and can take ~minutes per response. Caller can
# override via the timeout kwarg.
DEFAULT_CALL_TIMEOUT_SECONDS = 600.0


def _combine_prompt(system: str, user: str) -> str:
    """Merge system + user into a single prompt for CLIs without separate system flags.

    The XML-ish framing is robust across providers and models; CLIs that have
    a native --system-prompt flag (claude) override this in their subclass.
    """
    return (
        "<system_instructions>\n"
        f"{system}\n"
        "</system_instructions>\n\n"
        "<user_request>\n"
        f"{user}\n"
        "</user_request>"
    )


class LLMClient(ABC):
    """Abstract single-shot LLM caller. Each call() is independent — no history."""

    name: str = "abstract"

    @abstractmethod
    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Return the assistant text. May raise RuntimeError on CLI failure."""

    def edit_files(
        self,
        system: str,
        user: str,
        *,
        workdir: Path,
        timeout: float | None = None,
    ) -> str:
        """Run an agentic editing session in workdir.

        Subclasses override this when their CLI can safely edit files. Keeping
        this separate from call() preserves single-shot text calls for critic
        and other non-mutating roles.
        """
        raise NotImplementedError(f"{self.name} client does not support file-edit authoring")


def _run(cmd: list[str], *, timeout: float, stdin: str | None = None, cwd: Path | None = None) -> str:
    """Run a subprocess; return stdout; raise RuntimeError with full context on failure."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"LLM CLI call timed out after {timeout}s: {' '.join(cmd[:3])}..."
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"LLM CLI failed (exit {proc.returncode}): {' '.join(cmd[:3])}...\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    return proc.stdout


class HermesCliClient(LLMClient):
    """`hermes chat -q PROMPT [-m MODEL] [--provider P]`.

    Hermes is the repo's existing autoresearch driver; matches what
    student_finetune/run_v2.sh already uses. Default provider 'auto' lets
    Hermes pick.
    """

    name = "hermes"

    def __init__(self, *, model: str | None = None, provider: str = "auto") -> None:
        self.model = model
        self.provider = provider

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        prompt = _combine_prompt(system, user)
        cmd = [
            "hermes", "chat",
            "-q", prompt,
            "--yolo",            # skip interactive confirms
            "--max-turns", "1",  # single-shot — no agentic looping inside hermes
            "--ignore-rules",    # don't apply user-config rules to autoreason
            "-Q",                # quiet (suppress banner / status output)
        ]
        if self.model is not None:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS).strip()

    def edit_files(
        self,
        system: str,
        user: str,
        *,
        workdir: Path,
        timeout: float | None = None,
    ) -> str:
        prompt = _combine_prompt(system, user)
        cmd = [
            "hermes", "chat",
            "-q", prompt,
            "--yolo",
            "--max-turns", "40",
            "--ignore-rules",
            "-Q",
            "-t", "file,terminal",
        ]
        if self.model is not None:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS, cwd=workdir).strip()


class ClaudeCliClient(LLMClient):
    """`claude -p ... --system-prompt ...`.

    Uses Claude Code's `--print` non-interactive mode with a separate
    --system-prompt flag (so the system text is properly framed, not
    merged into user input).
    """

    name = "claude"

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        cmd = [
            "claude",
            "-p", user,
            "--system-prompt", system,
            "--bare",                              # minimal mode, no auto-memory etc.
            "--dangerously-skip-permissions",      # autoreason runs unattended
        ]
        if self.model is not None:
            cmd += ["--model", self.model]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS).strip()

    def edit_files(
        self,
        system: str,
        user: str,
        *,
        workdir: Path,
        timeout: float | None = None,
    ) -> str:
        cmd = [
            "claude",
            "-p", user,
            "--system-prompt", system,
            "--bare",
            "--dangerously-skip-permissions",
        ]
        if self.model is not None:
            cmd += ["--model", self.model]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS, cwd=workdir).strip()


class CodexCliClient(LLMClient):
    """`codex exec [PROMPT] [--config model=NAME]`.

    Codex doesn't have a separate --system-prompt; we combine system + user
    via _combine_prompt() and pass as the single argument. Model selection
    via -c model=... config override.
    """

    name = "codex"

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        prompt = _combine_prompt(system, user)
        cmd = ["codex", "exec"]
        if self.model is not None:
            cmd += ["-c", f"model={self.model}"]
        cmd += [prompt]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS).strip()

    def edit_files(
        self,
        system: str,
        user: str,
        *,
        workdir: Path,
        timeout: float | None = None,
    ) -> str:
        prompt = _combine_prompt(system, user)
        cmd = [
            "codex", "exec",
            "-C", str(workdir),
            "--sandbox", "workspace-write",
            "--full-auto",
        ]
        if self.model is not None:
            cmd += ["-c", f"model={self.model}"]
        cmd += [prompt]
        return _run(cmd, timeout=timeout or DEFAULT_CALL_TIMEOUT_SECONDS).strip()


_REGISTRY: dict[str, type[LLMClient]] = {
    "hermes": HermesCliClient,
    "claude": ClaudeCliClient,
    "codex":  CodexCliClient,
}


def make_llm_client(
    cli: LlmCliName | None = None,
    *,
    model: str | None = None,
    provider: str = "auto",
) -> LLMClient:
    """Factory: pick a CLI by name, env, or fall back to the default.

    Selection order:
      1. explicit `cli` arg
      2. AUTORESEARCH_LLM_CLI environment variable
      3. DEFAULT_CLI ("hermes")

    `provider` is Hermes-specific; ignored by Claude / Codex clients.
    """
    chosen = cli or os.environ.get("AUTORESEARCH_LLM_CLI") or DEFAULT_CLI
    if chosen not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM CLI: {chosen!r}. "
            f"Supported: {sorted(_REGISTRY)}."
        )
    cls = _REGISTRY[chosen]
    if cls is HermesCliClient:
        return HermesCliClient(model=model, provider=provider)
    return cls(model=model)


# Backwards-compat alias — older code imports `AgentClient`. Now an alias for
# the factory so the existing imports keep working without churn.
AgentClient = make_llm_client
