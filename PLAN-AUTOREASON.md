# Implementation Plan: Autoreason loop for ML training (issue #11)

Companion to [SPEC-AUTOREASON.md](SPEC-AUTOREASON.md). Plans the work into vertical slices; each slice delivers something testable on its own. Goal: one PR with multi-commit history, ≤ 8 tasks, each ≤ 5 files.

## Overview

We have the **evaluation/execution layer** (research_loop from PR #8). We need the **generation layer**: 3 LLM agents (Critic / Author B / Synthesizer) plus an orchestrator loop that runs them, applies their patches, runs training under a time budget, and converges via objective ML metrics. After this PR, `tournament autoreason --target student_v2` runs autonomously to convergence.

## Architecture Decisions

- **Single LLM model (Sonnet 4.6) for all 3 roles** — start simple; per-role tier can come later if cost matters.
- **Anthropic SDK directly, no litellm** — fewer indirections, prompt-cache friendly.
- **Patches via `git apply` / `git apply -R`** — fully reversible, every pass leaves working tree clean. Revert is the single source of truth for "did this round happen safely".
- **No LLM judges** — objective ML metrics through the existing `promote.decide()`. The paper's Borda is for subjective domains; we have measurable outcomes.
- **Time budget enforced at the adapter layer** — subprocess SIGTERM + 30s grace + SIGKILL. Recovery via `metrics_progress_v2.json` written by the trainer after each eval cycle.
- **Convergence: k=2 consecutive A wins** — paper's calibrated default; max-passes 15 ceiling.
- **Audit trail: every LLM call's full text persisted** to `history.jsonl` as `record_type ∈ {"critique", "patch_proposal", "synthesis"}`. Replayable post-hoc.
- **V2-only safety reused** — Author B's prompt forbids V1 edits; patch applicator double-checks before `git apply`. Belt and suspenders.

## Dependency Graph

```
metrics_progress writes (B0)         git apply/revert (P0)
       │                                     │
       ▼                                     ▼
adapter.train(max_seconds) (B1) ─────► PatchApplicator (P1)
       │                                     │
       └──────────────► AgentClient (A1) ◄──┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
        CriticAgent     AuthorBAgent     SynthesizerAgent
           (A2)              (A3)              (A4)
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                     Convergence loop (L1)
                              │
                              ▼
                  `tournament autoreason` CLI (L2)
                              │
                              ▼
                Mocked end-to-end integration (V1)
                              │
                              ▼
                  Manual smoke with real LLM (V2)
```

## Task List

### Phase 1: Primitives (foundation)

#### Task 1: Trainers write `metrics_progress_v2.json` after each eval

Both `student_finetune/train_v2.py` and `dino_finetune/train_dino_v2.py` already evaluate every epoch. Add a small write of `metrics_progress_v2.json` (same schema as `metrics_final_v2.json` plus `epochs_completed` and `is_partial=true`). Lets the time-budget recovery path read partial state without race conditions.

- **Acceptance:**
  - [ ] `metrics_progress_v2.json` written atomically after every eval (write-to-tmp + rename)
  - [ ] Schema matches `metrics_final_v2.json` keys, plus `is_partial: bool`, `epochs_completed: int`
  - [ ] V1 trainer behavior unchanged
- **Verification:** existing `test_productness_integration_smoke.py` still passes; new `test_metrics_progress_write.py` confirms file written + schema correct
- **Files:** `student_finetune/train_v2.py`, `dino_finetune/train_dino_v2.py`, `student_finetune/tests/test_metrics_progress_write.py`
- **Dependencies:** none — additive
- **Scope:** S

#### Task 2: Time-budget primitive in `adapter.train()`

`adapter.train(candidate, max_seconds=N)` enforces wall-clock; SIGTERM, 30s grace, SIGKILL. On timeout, parse `metrics_progress_v2.json` if present; status becomes `"timeout"`.

- **Acceptance:**
  - [ ] Adapter kills overrunning subprocess within 60s of `max_seconds`
  - [ ] Returns `TrainOutcome(status="timeout", metrics={partial}, return_code=-9 or similar)`
  - [ ] `promote.decide()` rejects timeout candidates regardless of partial metrics
- **Verification:** `pytest research_loop/tests/test_budget.py` (stub trainer that sleeps) — green; existing tests still pass
- **Files:** `research_loop/targets/_base.py`, `research_loop/promote.py`, `research_loop/tests/test_budget.py`
- **Dependencies:** Task 1 (for partial metrics recovery)
- **Scope:** M (3 files)

#### Task 3: Patch applicator (`research_loop/patch.py`)

Context manager that applies a unified diff via `git apply`, reverts on exit. Refuses any patch touching `V1_FORBIDDEN_PATHS`. Validates with `git apply --check` before apply.

- **Acceptance:**
  - [ ] `apply_patch(diff, repo)` is a `@contextmanager`; revert runs in `finally`
  - [ ] V1-touching patches raise `ValueError` before any filesystem mutation
  - [ ] Malformed patches raise `subprocess.CalledProcessError` (caught upstream)
- **Verification:** `pytest research_loop/tests/test_patch_applicator.py` — apply/revert roundtrip on tmp git repo, V1 refusal, malformed diff handling
- **Files:** `research_loop/patch.py`, `research_loop/tests/test_patch_applicator.py`
- **Dependencies:** none
- **Scope:** S (2 files)

### Checkpoint A: Primitives (after T1–T3)

- [ ] All three primitives pass their unit tests
- [ ] Existing 103 PR-#8 tests still green
- [ ] `git status` clean after `pytest` runs (revert always reverts)

---

### Phase 2: Agents (generation layer)

#### Task 4: `AgentClient` wrapper (`research_loop/agents/client.py`)

Thin Anthropic SDK wrapper. Reads API key from env, falls back to `~/.hermes/.env`. Single shared instance per process. Prompt caching enabled on system prompts for cross-call reuse within a pass. No retries inside the wrapper — let upstream handle.

- **Acceptance:**
  - [ ] `AgentClient.call(system, user, *, temperature)` returns the raw assistant text
  - [ ] Uses `cache_control={"type": "ephemeral"}` on the system block
  - [ ] Raises clear error when API key missing
- **Verification:** `pytest research_loop/tests/test_agent_client.py` — mocks `anthropic.Anthropic`, asserts the request shape (system block has cache_control, user message correct, no message history persisted across calls)
- **Files:** `research_loop/agents/__init__.py`, `research_loop/agents/client.py`, `research_loop/tests/test_agent_client.py`
- **Dependencies:** none (just adds `anthropic` to pyproject.toml deps)
- **Scope:** S

#### Task 5: `CriticAgent`, `AuthorBAgent`, `SynthesizerAgent`

Three `dataclass`-result agents. Each has a fixed system prompt (paper-style), a `render()` for the user message, and a `parse()` that turns LLM output into a typed object (`Critique` / `PatchProposal` / `Synthesis`). All three persist `raw` field for audit.

- **Acceptance:**
  - [ ] `Critique(summary, problems, raw)` from CriticAgent
  - [ ] `PatchProposal(rationale, diff, raw)` from AuthorBAgent — `diff` is a unified-diff string
  - [ ] `Synthesis(rationale, diff, raw)` from SynthesizerAgent — same format as PatchProposal
  - [ ] Author B's user message includes critic output and the trainer source; system prompt forbids V1 edits explicitly
  - [ ] Synthesizer sees only A and B's patches with anonymized labels (no metric history)
- **Verification:** `pytest research_loop/tests/test_agents.py` — `monkeypatch` `AgentClient.call` to canned responses, assert parsing produces typed objects with expected fields
- **Files:** `research_loop/agents/critic.py`, `research_loop/agents/author.py`, `research_loop/agents/synthesizer.py`, `research_loop/tests/test_agents.py`
- **Dependencies:** Task 4 (AgentClient)
- **Scope:** M (4 files)

### Checkpoint B: Agents (after T4–T5)

- [ ] Agents pass parsing tests with mocked LLM
- [ ] Patches produced by AuthorB/Synthesizer in tests pass `git apply --check`
- [ ] Each agent's call is fresh (no message history mutation between calls in the same process)

---

### Phase 3: Orchestration

#### Task 6: Convergence loop in `tournament.py`

Adds `tournament autoreason` subcommand. Each pass: Critic → Author B → Synthesizer → apply patches in turn (each in its own `apply_patch` context) → adapter.train under budget → promote.decide → write Decision. Track consecutive A wins; terminate at `k=2` or `--max-passes`.

Persists each LLM call's raw text into `history.jsonl` as `record_type ∈ {"critique", "patch_proposal", "synthesis"}` (extends `Candidate / Outcome / Decision` discriminated union).

- **Acceptance:**
  - [ ] `tournament autoreason --target student_v2 --max-passes N --convergence k --max-seconds-per-candidate S` end-to-end
  - [ ] Working tree clean after every pass (revert in `finally`)
  - [ ] Convergence at k consecutive A wins → exit
  - [ ] Hitting max-passes prints clear "did not converge" + exits with non-zero code
  - [ ] Author/Synthesizer patches recorded as candidates with `kind ∈ {"B", "AB"}`
- **Verification:** `pytest research_loop/tests/test_autoreason_loop.py` with mocked AgentClient + stub adapter — convergence test, max-passes test, working-tree-clean assertion
- **Files:** `research_loop/tournament.py`, `research_loop/candidate.py` (extend record types), `research_loop/tests/test_autoreason_loop.py`
- **Dependencies:** Tasks 1–5
- **Scope:** M (3 files)

### Checkpoint C: Orchestration (after T6)

- [ ] `tournament autoreason --dry-run` runs end-to-end on CPU
- [ ] Synthetic 5-pass scenario reaches convergence at k=2
- [ ] Total LLM calls per pass ≤ 5 (no judge calls)
- [ ] All previous tests still green

---

### Phase 4: Polish

#### Task 7: Audit trail + HANDOFF.md update

Make sure every LLM call writes its raw text into `history.jsonl` (not just the parsed dataclass). Update `HANDOFF.md` with the new "Stage 1 / Stage 2 via `tournament autoreason`" section, replacing the prior "manual `run-round --baseline-only`" instructions.

- **Acceptance:**
  - [ ] Replaying a `history.jsonl` reconstructs every LLM call (raw text preserved verbatim)
  - [ ] HANDOFF.md describes how to launch the autonomous loop and how to interpret a run
- **Verification:** unit test reads a mock history.jsonl and asserts all expected `record_type` values present; HANDOFF reads cleanly to a new engineer
- **Files:** `research_loop/tournament.py` (audit hooks), `HANDOFF.md`
- **Dependencies:** Task 6
- **Scope:** S (2 files)

#### Task 8: Real-LLM smoke (manual, not in CI)

One pass against the current `train_v2.py` with a real Anthropic API key, but `--dry-run-train` (skip GPU). Confirms: prompt → critique → diff → applies cleanly via `git apply --check` → reverts. Document the run in HANDOFF.md as a one-time manual verification.

- **Acceptance:**
  - [ ] Real Sonnet 4.6 call returns a parseable critique
  - [ ] Author B's diff applies via `git apply --check` without errors
  - [ ] Synthesizer's diff also applies cleanly
  - [ ] Working tree clean after revert
- **Verification:** manual run + record output snippet in HANDOFF.md
- **Files:** `HANDOFF.md` only (records the smoke evidence)
- **Dependencies:** Task 6
- **Scope:** XS

### Checkpoint D: Complete

- [ ] All tasks T1–T8 complete
- [ ] `pytest -q` 100% green
- [ ] Manual smoke (T8) recorded
- [ ] Single PR with 7-8 atomic commits ready for review

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM produces unapplyable diff | High | `git apply --check` before apply; malformed → that candidate skipped, not whole loop |
| LLM tries to edit V1 file | High | Author B system prompt forbids; patch applicator double-checks `V1_FORBIDDEN_PATHS` before `git apply` |
| Subprocess SIGTERM doesn't unblock CUDA | Medium | 30s grace then SIGKILL; CUDA process gets killed regardless |
| Working tree dirty before pass | High | Tournament refuses to start unless `git status --porcelain` is empty (operator must commit/stash) |
| Cost runaway (LLM calls × passes) | Medium | Hard caps: max-passes 15, ≤ 5 calls/pass, ≤ $5/run on Sonnet 4.6 |
| Cache-pollution between passes | Medium | Adapter-sha cache keying (already shipped in PR #8) — patches that change LoRA training change the sha automatically |
| Convergence never reached | Low | max-passes ceiling exits with non-zero; operator inspects history.jsonl |
| LLM hallucinates a metric improvement | High | objective `promote.decide()` is the only path to "winner" — LLM cannot vote |

## Open Questions

(Resolved per user direction 2026-04-25:)
- Patch path style: `a/student_finetune/train_v2.py` (repo-rooted) ✅
- Malformed diff: skip that candidate, round continues ✅
- Synthesizer sees only patches, no metrics ✅

No further open questions before implementation.

## Parallelization

All tasks have linear dependencies (T1 → T2 ... → T6 → T7 → T8). Single-agent linear implementation is fine; can't meaningfully parallelize without splitting between humans, which is not the case here.
