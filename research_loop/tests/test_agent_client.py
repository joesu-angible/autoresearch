"""T4 verification: AgentClient wrapper.

No real Anthropic API call — we mock at the SDK boundary. Verifies:
  - request shape (system block has cache_control: ephemeral)
  - no message history persisted across calls (fresh-agent invariant)
  - API key resolves from env, then from ~/.hermes/.env
  - missing key raises a clear error
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_loop.agents.client import (
    AgentClient,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    resolve_api_key,
)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert resolve_api_key() == "sk-from-env"


def test_resolve_api_key_falls_back_to_hermes_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".hermes").mkdir()
    (fake_home / ".hermes" / ".env").write_text(
        "# comment line\n"
        "ANTHROPIC_API_KEY=sk-from-hermes\n"
        "OTHER=ignored\n"
    )
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert resolve_api_key() == "sk-from-hermes"


def test_resolve_api_key_handles_quoted_values(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".hermes").mkdir()
    (fake_home / ".hermes" / ".env").write_text('ANTHROPIC_API_KEY="sk-quoted"\n')
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert resolve_api_key() == "sk-quoted"


def test_resolve_api_key_missing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no .hermes/.env
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not found"):
        resolve_api_key()


# ---------------------------------------------------------------------------
# AgentClient request shape
# ---------------------------------------------------------------------------

@patch("research_loop.agents.client.anthropic.Anthropic")
def test_call_sends_system_block_with_cache_control(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="response text")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    client = AgentClient(api_key="sk-test")
    out = client.call("system text", "user text", temperature=0.5)

    assert out == "response text"
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL
    assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call_kwargs["temperature"] == 0.5
    # System block: list with one text block + cache_control ephemeral
    sys_block = call_kwargs["system"]
    assert isinstance(sys_block, list) and len(sys_block) == 1
    assert sys_block[0]["type"] == "text"
    assert sys_block[0]["text"] == "system text"
    assert sys_block[0]["cache_control"] == {"type": "ephemeral"}
    # Messages: just the one user turn — no chat history
    assert call_kwargs["messages"] == [{"role": "user", "content": "user text"}]


@patch("research_loop.agents.client.anthropic.Anthropic")
def test_call_does_not_persist_history_across_calls(mock_anthropic_cls):
    """Fresh-agent invariant: each call sends only its own user turn,
    not accumulated history from prior calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="r")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    client = AgentClient(api_key="sk-test")
    client.call("sys", "user1")
    client.call("sys", "user2")
    client.call("sys", "user3")

    # Each create() call should have exactly ONE message — the current user turn
    for call in mock_client.messages.create.call_args_list:
        assert len(call.kwargs["messages"]) == 1


@patch("research_loop.agents.client.anthropic.Anthropic")
def test_call_max_tokens_override(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="r")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    client = AgentClient(api_key="sk-test", max_tokens=2048)
    client.call("sys", "user", max_tokens=8192)
    assert mock_client.messages.create.call_args.kwargs["max_tokens"] == 8192


@patch("research_loop.agents.client.anthropic.Anthropic")
def test_default_model_is_sonnet_4_6(mock_anthropic_cls):
    """SPEC §Tech stack pins Sonnet 4.6 for all three agent roles."""
    assert DEFAULT_MODEL == "claude-sonnet-4-6"
