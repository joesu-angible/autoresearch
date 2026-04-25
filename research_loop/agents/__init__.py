"""LLM agent layer for the autoreason loop.

Three roles, all separate single-shot calls (the autoreason "fresh agent"
invariant — no shared message history between the Critic, Author B, and
Synthesizer):

  - critic.CriticAgent       — finds problems in the incumbent A
  - author.AuthorBAgent      — produces a unified-diff patch for B
  - synthesizer.SynthesizerAgent — produces a synthesis patch (AB)

All three share a single AgentClient (Anthropic SDK wrapper with prompt
caching enabled on system prompts) but never share messages.
"""

from research_loop.agents.client import AgentClient

__all__ = ["AgentClient"]
