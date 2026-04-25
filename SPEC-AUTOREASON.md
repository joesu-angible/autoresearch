# Spec: Autoreason loop for ML training (issue #11)

Tracks GitHub issue [#11](https://github.com/joesu-angible/issues/11). Closes #10 as a sub-component.

**Branch:** `feat/autoreason-llm-loop`

## Objective

The research_loop scaffold from PR #8 is the **evaluation/execution layer** of an autoreason-style tournament — Candidate/Outcome/Decision schema, V2-only target adapters, productness-aware promotion guardrails. What's missing is the **generation layer**: the autonomous LLM-driven proposer that makes autoreason actually autoreason. Today every B/AB candidate's patch is a `# TODO` placeholder a human fills in.

This spec adds the generation layer. After this PR, `tournament autoreason --target student_v2 --max-passes 15` runs end-to-end without human intervention:

```
Pass 1:
  A = current train_v2.py
  Critic LLM ─→ critique
  Author B LLM ─→ patch B (unified diff)
  Synthesizer LLM ─→ patch AB (unified diff)
  Apply patches in temp branches → run training (subject to per-candidate time budget)
  promote.decide() over objective metrics → winner
  → Winner becomes new A
Pass 2: ... (until A wins k=2 consecutive, or max-passes reached)
```

Success = a `tournament autoreason` invocation can autonomously drive a multi-pass refinement loop on `train_v2.py` or `train_dino_v2.py`, terminate cleanly at convergence, and leave the working tree clean.

## Tech stack

- **LLM access**: subprocess to a local CLI — `hermes`, `claude`, or `codex`. **No raw API keys in this repo.** Auth and rate limits stay in whichever tool the operator already runs. Selection via `--llm-cli` flag or `AUTORESEARCH_LLM_CLI` env; default `hermes` (matches the existing `student_finetune/run_v2.sh` convention).
- **Model passthrough**: `--llm-model` flag forwarded to whichever CLI is selected (e.g. `anthropic/claude-sonnet-4` for hermes, `claude-sonnet-4-6` for claude, `gpt-5` for codex).
- **Patch format**: unified diff applied via `subprocess.run(["git", "apply", ...])`. Revert via `git apply -R`.
- **Existing**: Python 3.10+, PyTorch, the research_loop infrastructure already in master.

No new pip dependencies — autoreason runs entirely on already-installed CLIs.

## Commands

```bash
# Run the full autoreason loop on student_v2 (autonomous; ~hours per pass)
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 \
    --max-passes 15 \
    --convergence 2 \
    --max-seconds-per-candidate 9000 \
    --hypothesis-seed "close productness_neg_acc gap below 0.85"

# Single-pass dry-run with mocked LLM (fast, for tests)
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 --max-passes 1 --dry-run

# Existing commands still work (propose / rank / run / promote / run-round)
.venv/bin/python -m research_loop.tournament run-round --target student_v2 --baseline-only

# Tests (CPU-only, mocked LLM + mocked subprocess)
.venv/bin/python -m pytest research_loop/tests dino_finetune/tests student_finetune/tests/test_*productness*.py student_finetune/tests/test_adapter_sha.py -q
```

## Project structure

```
autoresearch/
├── research_loop/
│   ├── agents/                          # NEW — generation layer
│   │   ├── __init__.py
│   │   ├── client.py                    # Anthropic SDK wrapper, env-var key, retries
│   │   ├── critic.py                    # CriticAgent: train_v2.py + history → Critique
│   │   ├── author.py                    # AuthorBAgent: critique + train_v2.py → unified diff
│   │   └── synthesizer.py               # SynthesizerAgent: A + B's patch → AB patch
│   ├── patch.py                         # NEW — apply / revert unified diff via git apply
│   ├── budget.py                        # NEW — wall-clock timeout primitive (was issue #10)
│   ├── candidate.py                     # extended: PatchProposal, Critique dataclasses
│   ├── promote.py                       # extended: timeout outcomes auto-rejected
│   ├── targets/_base.py                 # extended: train(max_seconds=N) honors budget
│   ├── tournament.py                    # extended: `autoreason` subcommand + convergence loop
│   └── tests/
│       ├── test_agents.py               # mocked LLM responses + parsing
│       ├── test_patch_applicator.py     # apply/revert roundtrip
│       ├── test_budget.py               # SIGTERM kill + timeout outcome
│       └── test_autoreason_loop.py      # full convergence flow with stub adapters
└── student_finetune/train_v2.py         # extended: writes metrics_progress_v2.json after each eval
└── dino_finetune/train_dino_v2.py       # same
```

## Code style

```python
# research_loop/agents/client.py — thin wrapper, retries, prompt-cached system prompt
class AgentClient:
    """Single shared Anthropic SDK client with prompt caching enabled.

    Uses ephemeral cache breakpoints on system prompts so the Critic / Author /
    Synthesizer system messages get cached across calls within a pass. Each
    role reuses the same client; fresh user-message context per call ensures
    no shared chat history (the autoreason 'fresh agent' invariant).
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 4096):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str, *, temperature: float = 0.8) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.content[0].text


# research_loop/agents/critic.py — typed, returns structured Critique
@dataclass(frozen=True)
class Critique:
    summary: str
    problems: list[str]      # bullet list — concrete, no fixes proposed
    raw: str                 # full LLM text for audit log

class CriticAgent:
    SYSTEM_PROMPT = """You are reviewing a machine-learning training script and
    its recent experiment history. Your only job is to identify concrete problems —
    do not propose fixes. Be specific. Reference exact file:line locations or
    metric values. If the run is healthy, say so explicitly."""

    def __init__(self, client: AgentClient): self.client = client

    def critique(self, trainer_src: str, history: list[Outcome],
                 results_tsv_tail: str) -> Critique:
        user = self._render_prompt(trainer_src, history, results_tsv_tail)
        raw = self.client.call(self.SYSTEM_PROMPT, user, temperature=0.3)
        return self._parse(raw)
```

Patches: unified diff produced by `Author B`, validated for V2 safety, applied with `git apply --check` first, then `git apply`. Revert via `git apply -R`.

```python
# research_loop/patch.py
@contextmanager
def apply_patch(patch_text: str, repo: Path):
    """Apply unified diff; revert on context exit. Raises if patch touches V1 files."""
    if _touches_v1(patch_text):
        raise ValueError("Patch touches a V1_FORBIDDEN_PATHS file")
    _git("apply", "--check", input=patch_text, cwd=repo)
    _git("apply", input=patch_text, cwd=repo)
    try:
        yield
    finally:
        _git("apply", "-R", input=patch_text, cwd=repo)
```

## Testing strategy

CPU-only unit + integration tests. No real GPU, no real LLM call in the test suite.

- `research_loop/tests/test_agents.py` — Critic/Author/Synthesizer with `monkeypatch` on `AgentClient.call` returning canned strings. Verify parsing into typed objects, fresh-agent invariant (each `.critique()` call uses a new system+user pair, no message history mutation).
- `research_loop/tests/test_patch_applicator.py` — apply/revert roundtrip on a tmp git repo (fixture); refusal when patch references `V1_FORBIDDEN_PATHS`.
- `research_loop/tests/test_budget.py` — `adapter.train(max_seconds=2)` against a stub trainer that sleeps 5s; assert subprocess killed, status=`"timeout"`, partial metrics parsed if present.
- `research_loop/tests/test_autoreason_loop.py` — full convergence with mocked LLM + stub adapter that returns deterministic Outcomes. Cover:
  - convergence at k=2 (A wins twice → done)
  - max-passes ceiling
  - timeout candidate auto-rejected
  - patch revert on round end (working tree clean after each pass)

Coverage target: new code ≥ 80% line coverage.

## Boundaries

**Always:**
- Use `anthropic.Anthropic()` for LLM calls; respect existing repo env conventions.
- Apply patches via `git apply` and revert at the end of every pass — working tree must be the same after a pass as before.
- Refuse any patch that touches `V1_FORBIDDEN_PATHS` (LLM may try; double-check after generation).
- Write each LLM call's `raw` text into `research_loop/history.jsonl` for audit (`kind="critique"`, `"patch_proposal"`, `"synthesis"`).
- Cap `max_seconds_per_candidate` even for production runs — autoreason without timeout could chew 50h on one bad LLM-generated patch.

**Ask first:**
- Adding any LLM provider beyond Anthropic SDK.
- Increasing `max_passes` above 30.
- Switching to LLM-as-judge (paper says don't for code/objective domains).
- Allowing autoreason to write into V1 files.
- Adding patches that introduce new files (Author B should be modifying existing trainers, not creating new modules).

**Never:**
- Commit the LLM `raw` text containing API keys or secrets.
- Persist Anthropic API key in any committed file.
- Run autoreason on a dirty working tree (we'd lose the user's uncommitted work).
- Use `--no-verify` to bypass tests.
- Skip the patch revert step — even on error paths, the `try/finally` must run.

## Success criteria

1. `tournament autoreason --target student_v2 --max-passes 1 --dry-run` runs in CI/CPU without launching real subprocess or hitting real LLM API.
2. `test_autoreason_loop.py` proves a synthetic 5-pass run converges at k=2 and produces clean working tree after each pass.
3. With real LLM (manual smoke), one pass against the current `train_v2.py` produces a valid Critique → unified-diff PatchProposal → unified-diff Synthesis. Patches apply via `git apply --check` without errors.
4. Time budget: `adapter.train(max_seconds=N)` kills overrunning subprocess within 60s of expiry, returns `Outcome(status="timeout")`. `promote.decide()` rejects timeout candidates regardless of partial metrics.
5. V2-only safety preserved end-to-end: a synthetic patch that edits `student_finetune/train.py` (V1) is rejected by `apply_patch` before `git apply` runs.
6. Audit trail: after a 5-pass run on the synthetic test, `research_loop/history.jsonl` contains 5 critique records + 5 patch_proposal + 5 synthesis + 5×3 outcomes + 5 decisions = 40 records minimum, all replayable.
7. Total LLM call count per pass ≤ 5 (1 Critic + 1 Author B + 1 Synthesizer + 0 judges, with retries counted separately).
8. PR is single-branch, multi-commit (sub-tasks A–E from issue #11 land as separate atomic commits), 100% green tests on master at PR open time.

## Open questions

1. **Patch base directory**: Author B sees `student_finetune/train_v2.py` as input — should the unified diff use `a/student_finetune/train_v2.py` paths (repo-rooted) or just `a/train_v2.py`? Standard `git apply` from repo root expects the former. **Default assumption: repo-rooted paths.**
2. **What if the LLM produces a malformed diff?** Today: error → that candidate skipped, round continues with whoever else has valid patches. If both B and AB malformed, A wins by default. **Default assumption: be lenient — single failure does not abort the loop.**
3. **Should the Synthesizer see metric history when synthesizing AB?** Paper says synthesizer gets A + B with anonymized labels and "no drafting history." For ML, leaning toward keeping it pure — only the patches, not the metrics, to avoid the synthesizer over-fitting to one data point. **Default assumption: synthesizer sees A and B's patch only, no metrics.**

→ Confirm or correct these and I'll move to the plan phase.
