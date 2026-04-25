"""T4 verification: CLI-backed LLM clients (hermes / claude / codex).

No real CLI invocation — we mock subprocess.run at the boundary. Verifies:
  - factory dispatch by name + AUTORESEARCH_LLM_CLI env override
  - each CLI's command-line shape (correct flags, model passthrough)
  - fresh-agent invariant (each .call() is one independent subprocess)
  - timeout / non-zero exit are surfaced as clear RuntimeErrors
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from research_loop.agents.client import (
    DEFAULT_CLI,
    ClaudeCliClient,
    CodexCliClient,
    HermesCliClient,
    make_llm_client,
)


def _ok(stdout: str, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------

def test_default_cli_is_hermes():
    assert DEFAULT_CLI == "hermes"


def test_factory_uses_explicit_arg(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_LLM_CLI", raising=False)
    assert isinstance(make_llm_client("claude"), ClaudeCliClient)
    assert isinstance(make_llm_client("codex"), CodexCliClient)
    assert isinstance(make_llm_client("hermes"), HermesCliClient)


def test_factory_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_LLM_CLI", "claude")
    assert isinstance(make_llm_client(None), ClaudeCliClient)


def test_factory_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_LLM_CLI", raising=False)
    assert isinstance(make_llm_client(None), HermesCliClient)


def test_factory_rejects_unknown_cli(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_LLM_CLI", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM CLI"):
        make_llm_client("totally-bogus")


def test_factory_threads_model_to_hermes():
    client = make_llm_client("hermes", model="anthropic/claude-sonnet-4")
    assert isinstance(client, HermesCliClient)
    assert client.model == "anthropic/claude-sonnet-4"


def test_factory_threads_model_to_claude():
    client = make_llm_client("claude", model="claude-sonnet-4-6")
    assert client.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# HermesCliClient
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.subprocess.run")
def test_hermes_call_passes_combined_prompt_and_flags(mock_run):
    mock_run.return_value = _ok("hermes response text\n")
    client = HermesCliClient(model="anthropic/claude-sonnet-4", provider="anthropic")
    out = client.call("system text", "user text")

    assert out == "hermes response text"
    cmd = mock_run.call_args.args[0]
    assert cmd[0:2] == ["hermes", "chat"]
    assert "-q" in cmd
    prompt = cmd[cmd.index("-q") + 1]
    assert "system text" in prompt
    assert "user text" in prompt
    assert "--max-turns" in cmd and "1" == cmd[cmd.index("--max-turns") + 1]
    assert "-m" in cmd and "anthropic/claude-sonnet-4" == cmd[cmd.index("-m") + 1]
    assert "--provider" in cmd and "anthropic" == cmd[cmd.index("--provider") + 1]
    assert "--yolo" in cmd
    # No chat history input — single shot
    assert mock_run.call_args.kwargs.get("input") is None


@patch("research_loop.agents.client.subprocess.run")
def test_hermes_call_without_model_omits_flag(mock_run):
    mock_run.return_value = _ok("text")
    client = HermesCliClient()
    client.call("sys", "user")
    cmd = mock_run.call_args.args[0]
    assert "-m" not in cmd  # no model override


# ---------------------------------------------------------------------------
# ClaudeCliClient
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.subprocess.run")
def test_claude_call_uses_native_system_prompt_flag(mock_run):
    mock_run.return_value = _ok("claude response\n")
    client = ClaudeCliClient(model="claude-sonnet-4-6")
    out = client.call("system text", "user text")

    assert out == "claude response"
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "claude"
    assert "-p" in cmd and "user text" == cmd[cmd.index("-p") + 1]
    assert "--system-prompt" in cmd and "system text" == cmd[cmd.index("--system-prompt") + 1]
    assert "--bare" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--model" in cmd and "claude-sonnet-4-6" == cmd[cmd.index("--model") + 1]


# ---------------------------------------------------------------------------
# CodexCliClient
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.subprocess.run")
def test_codex_call_uses_combined_prompt_and_config_override(mock_run):
    mock_run.return_value = _ok("codex response\n")
    client = CodexCliClient(model="gpt-5")
    out = client.call("system text", "user text")

    assert out == "codex response"
    cmd = mock_run.call_args.args[0]
    assert cmd[:2] == ["codex", "exec"]
    # Model passed via -c key=value override
    assert "-c" in cmd and "model=gpt-5" == cmd[cmd.index("-c") + 1]
    # Combined prompt is the trailing arg
    last = cmd[-1]
    assert "system text" in last
    assert "user text" in last


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.subprocess.run")
def test_non_zero_exit_raises_runtime_error_with_stderr(mock_run):
    proc = MagicMock()
    proc.returncode = 2
    proc.stdout = ""
    proc.stderr = "rate limit hit"
    mock_run.return_value = proc

    client = HermesCliClient()
    with pytest.raises(RuntimeError, match="rate limit hit"):
        client.call("sys", "user")


@patch("research_loop.agents.client.subprocess.run")
def test_timeout_raises_runtime_error(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["hermes"], timeout=5)
    client = HermesCliClient()
    with pytest.raises(RuntimeError, match="timed out"):
        client.call("sys", "user", timeout=5)


# ---------------------------------------------------------------------------
# Fresh-agent invariant across roles
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.subprocess.run")
def test_each_call_is_independent_subprocess(mock_run):
    """No shared state across .call() invocations — each is its own subprocess."""
    mock_run.return_value = _ok("response")
    client = HermesCliClient()

    client.call("sys1", "user1")
    client.call("sys2", "user2")
    client.call("sys3", "user3")

    assert mock_run.call_count == 3
    # Each call's prompt embeds its own system+user — no accumulated history
    for call, expected_sys, expected_user in zip(
        mock_run.call_args_list,
        ["sys1", "sys2", "sys3"],
        ["user1", "user2", "user3"],
    ):
        cmd = call.args[0]
        prompt = cmd[cmd.index("-q") + 1]
        assert expected_sys in prompt
        assert expected_user in prompt
