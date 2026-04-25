"""T5 verification: Critic / Author B / Synthesizer agents.

Mocked AgentClient.call — no real LLM hits. Verifies parsing, fresh-agent
invariant (each call is single-shot, no history mutation), V1-safety prompt
language, and that author/synthesizer outputs are valid unified diffs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_loop.agents.author import AuthorBAgent, PatchProposal, _parse_patch_proposal
from research_loop.agents.client import AgentClient
from research_loop.agents.critic import Critique, CriticAgent, _parse_critique
from research_loop.agents.synthesizer import SynthesizerAgent, _parse_synthesis


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

CRITIC_RESPONSE_SAMPLE = """# Summary
The student trainer is plateauing on combined around 0.79 with productness neg_acc stuck at 0.76 across the last three epochs.

# Problems
- The productness BCE weight may be too low to drive meaningful gradient through the head; pos_acc is already saturated near 1.0 while neg_acc has not improved since epoch 1.
- The strong-aug RandomErasing (p=0.2) may be erasing salient features in commodity images, given the unlabeled commodity ratio of 15%.
- No early stopping on productness_neg_acc — the run will burn 30 epochs even when retrieval has already converged.
"""


def test_parse_critique_extracts_summary_and_problems():
    critique = _parse_critique(CRITIC_RESPONSE_SAMPLE)
    assert "plateauing" in critique.summary.lower()
    assert len(critique.problems) == 3
    assert "productness BCE weight" in critique.problems[0]
    assert critique.raw == CRITIC_RESPONSE_SAMPLE


def test_parse_critique_no_actionable_problems():
    raw = (
        "# Summary\nRun is healthy.\n\n"
        "# Problems\nNo actionable problems — incumbent is performing within expectations."
    )
    critique = _parse_critique(raw)
    assert critique.summary == "Run is healthy."
    # Sentence is captured as one bullet-less line — that's fine; check raw is preserved
    assert critique.raw == raw


def test_critic_agent_calls_client_with_low_temperature():
    mock_client = MagicMock(spec=AgentClient)
    mock_client.call.return_value = CRITIC_RESPONSE_SAMPLE
    agent = CriticAgent(mock_client)
    out = agent.critique(
        trainer_source="def main(): pass",
        results_tsv_tail="round_id\tcombined\n",
        recent_outcomes_json='{"status":"success"}',
    )
    assert isinstance(out, Critique)
    assert mock_client.call.call_args.kwargs["temperature"] == 0.3
    # System prompt enforces critic posture: find problems, no fixes
    sys_arg = mock_client.call.call_args.args[0]
    sys_lower = sys_arg.lower()
    assert "identify concrete problems" in sys_lower
    assert "do not propose fixes" in sys_lower


# ---------------------------------------------------------------------------
# AuthorBAgent
# ---------------------------------------------------------------------------

AUTHOR_B_RESPONSE_SAMPLE = """# Rationale
Raise PRODUCTNESS_CLS_WEIGHT from 0.02 to 0.05 to give the productness head a stronger gradient signal addressing the plateau in neg_acc identified by the critic.

# Diff
```
diff --git a/student_finetune/train_v2.py b/student_finetune/train_v2.py
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -120,7 +120,7 @@
 USE_PRODUCTNESS_CLS = True
-PRODUCTNESS_CLS_WEIGHT = 0.02
+PRODUCTNESS_CLS_WEIGHT = 0.05
 PRODUCTNESS_HEAD_HIDDEN = 256
```
"""


def test_parse_patch_proposal_extracts_rationale_and_diff():
    p = _parse_patch_proposal(AUTHOR_B_RESPONSE_SAMPLE)
    assert "0.05" in p.rationale
    assert "diff --git" in p.diff
    assert "PRODUCTNESS_CLS_WEIGHT = 0.05" in p.diff
    assert p.raw == AUTHOR_B_RESPONSE_SAMPLE


def test_parse_patch_proposal_no_patch_sentinel():
    raw = (
        "# Rationale\nNo changes proposed.\n\n"
        "# Diff\nNO_PATCH"
    )
    p = _parse_patch_proposal(raw)
    assert p.diff == ""
    assert "No changes proposed" in p.rationale


def test_parse_patch_proposal_no_patch_in_fence():
    raw = (
        "# Rationale\nNothing to do.\n\n"
        "# Diff\n```\nNO_PATCH\n```"
    )
    p = _parse_patch_proposal(raw)
    assert p.diff == ""


def test_author_agent_system_prompt_forbids_v1_edits():
    """The Author B system prompt must explicitly list every V1_FORBIDDEN_PATHS entry."""
    from research_loop.agents.author import AUTHOR_B_SYSTEM_PROMPT
    from research_loop.targets._base import V1_FORBIDDEN_PATHS

    for path in V1_FORBIDDEN_PATHS:
        assert path in AUTHOR_B_SYSTEM_PROMPT, f"V1 path {path} not enumerated in author prompt"


def test_author_agent_calls_client_with_higher_temperature():
    mock_client = MagicMock(spec=AgentClient)
    mock_client.call.return_value = AUTHOR_B_RESPONSE_SAMPLE
    agent = AuthorBAgent(mock_client)
    out = agent.author(
        trainer_source="x = 1\n",
        trainer_path="student_finetune/train_v2.py",
        critique_text="problem 1",
    )
    assert isinstance(out, PatchProposal)
    assert mock_client.call.call_args.kwargs["temperature"] == 0.8


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

SYNTH_RESPONSE_SAMPLE = """# Rationale
Halved the proposed weight bump from 0.05 to 0.035, keeping the direction X identifies but committing less than its full step.

# Diff
```
diff --git a/student_finetune/train_v2.py b/student_finetune/train_v2.py
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -120,7 +120,7 @@
 USE_PRODUCTNESS_CLS = True
-PRODUCTNESS_CLS_WEIGHT = 0.02
+PRODUCTNESS_CLS_WEIGHT = 0.035
 PRODUCTNESS_HEAD_HIDDEN = 256
```
"""


def test_parse_synthesis_extracts_rationale_and_diff():
    p = _parse_synthesis(SYNTH_RESPONSE_SAMPLE)
    assert "halved" in p.rationale.lower() or "0.035" in p.rationale
    assert "PRODUCTNESS_CLS_WEIGHT = 0.035" in p.diff


def test_synthesizer_uses_anonymized_labels_in_user_message():
    """Per autoreason paper §2: synthesizer sees X / Y, not A / B."""
    mock_client = MagicMock(spec=AgentClient)
    mock_client.call.return_value = SYNTH_RESPONSE_SAMPLE
    agent = SynthesizerAgent(mock_client)
    agent.synthesize(
        patch_x="patch_x_text",
        patch_y="patch_y_text",
        trainer_path="student_finetune/train_v2.py",
    )
    user_message = mock_client.call.call_args.args[1]
    assert "## Patch X" in user_message
    assert "## Patch Y" in user_message
    # Verifies anonymization: must NOT leak "A" / "B" / "AB" labels
    assert "## Patch A" not in user_message
    assert "## Patch B" not in user_message


def test_synthesizer_does_not_see_metrics():
    """Per project decision 2026-04-25: synthesizer sees only patches, no metric history."""
    mock_client = MagicMock(spec=AgentClient)
    mock_client.call.return_value = SYNTH_RESPONSE_SAMPLE
    agent = SynthesizerAgent(mock_client)
    agent.synthesize(
        patch_x="patch_x", patch_y="patch_y",
        trainer_path="student_finetune/train_v2.py",
    )
    user_message = mock_client.call.call_args.args[1]
    # No metric keys should appear in the synthesizer's input
    for forbidden_metric_term in ("combined", "recall_1", "productness_neg_acc", "results_v2"):
        assert forbidden_metric_term not in user_message, \
            f"Synthesizer leaked metric input: '{forbidden_metric_term}'"


# ---------------------------------------------------------------------------
# Fresh-agent invariant (cross-cutting)
# ---------------------------------------------------------------------------

def test_three_agents_make_three_independent_calls():
    """All three roles share an AgentClient but never share message history.

    Each .critique() / .author() / .synthesize() call invokes
    AgentClient.call exactly once with a fresh user message.
    """
    mock_client = MagicMock(spec=AgentClient)
    mock_client.call.side_effect = [
        CRITIC_RESPONSE_SAMPLE,
        AUTHOR_B_RESPONSE_SAMPLE,
        SYNTH_RESPONSE_SAMPLE,
    ]
    critic = CriticAgent(mock_client)
    author = AuthorBAgent(mock_client)
    synth = SynthesizerAgent(mock_client)

    critic.critique(
        trainer_source="x = 1",
        results_tsv_tail="",
        recent_outcomes_json="",
    )
    author.author(
        trainer_source="x = 1",
        trainer_path="student_finetune/train_v2.py",
        critique_text="...",
    )
    synth.synthesize(
        patch_x="...", patch_y="...",
        trainer_path="student_finetune/train_v2.py",
    )

    # Three distinct calls, each with its own system prompt
    assert mock_client.call.call_count == 3
    system_prompts = [c.args[0] for c in mock_client.call.call_args_list]
    assert len(set(system_prompts)) == 3  # all three role prompts are distinct
