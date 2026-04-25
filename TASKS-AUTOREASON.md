# Tasks: Autoreason loop for ML training (issue #11)

Companion to [SPEC-AUTOREASON.md](SPEC-AUTOREASON.md) and [PLAN-AUTOREASON.md](PLAN-AUTOREASON.md).

8 tasks, ordered by dependency. Each becomes one atomic commit on `feat/autoreason-llm-loop`.

---

## Phase 1 — Primitives

- [ ] **T1. Trainers write `metrics_progress_v2.json` after each eval**
  - Acceptance: file written atomically (write-temp-then-rename) after every retrieval+productness eval cycle in both trainers; schema matches `metrics_final_v2.json` plus `is_partial=true` and `epochs_completed: int`.
  - Verify: `pytest student_finetune/tests/test_metrics_progress_write.py -q` passes; existing 103 tests stay green.
  - Files: `student_finetune/train_v2.py`, `dino_finetune/train_dino_v2.py`, `student_finetune/tests/test_metrics_progress_write.py`.
  - Commit message: `feat(trainers): write metrics_progress_v2.json after each eval for timeout recovery`

- [ ] **T2. `adapter.train(max_seconds=N)` time budget**
  - Acceptance: subprocess SIGTERM at budget, 30s grace, then SIGKILL; `TrainOutcome(status="timeout", metrics={partial})` returned; partial metrics parsed from `metrics_progress_v2.json` if present. `promote.decide()` rejects timeout candidates regardless of metric values.
  - Verify: `pytest research_loop/tests/test_budget.py -q`. Stub trainer that sleeps past `max_seconds`; confirm kill timing within 60s of expiry, status=`"timeout"`.
  - Files: `research_loop/targets/_base.py`, `research_loop/promote.py`, `research_loop/tests/test_budget.py`.
  - Commit message: `feat(research_loop): per-trial time budget via adapter.train(max_seconds=N)`

- [ ] **T3. `research_loop/patch.py` — apply / revert unified diffs**
  - Acceptance: `apply_patch(diff_text, repo)` is a `@contextmanager`; revert runs in `finally` even on exception; refuses any diff that touches `V1_FORBIDDEN_PATHS` (raises `ValueError` *before* any `git apply`); validates with `git apply --check` before apply.
  - Verify: `pytest research_loop/tests/test_patch_applicator.py -q` — apply/revert roundtrip on a tmp git repo fixture; V1 refusal; malformed diff handling; revert-on-exception path.
  - Files: `research_loop/patch.py`, `research_loop/tests/test_patch_applicator.py`.
  - Commit message: `feat(research_loop): patch applicator with V1 safety + revert-on-exit`

### ✅ Checkpoint A — Primitives complete

```
pytest research_loop/tests/test_budget.py \
       research_loop/tests/test_patch_applicator.py \
       student_finetune/tests/test_metrics_progress_write.py -q
```

All previous (PR #8) tests stay green. `git status` is clean after the suite (revert always runs).

---

## Phase 2 — Agents

- [ ] **T4. `AgentClient` Anthropic SDK wrapper**
  - Acceptance: `research_loop/agents/client.py` exposes `AgentClient(model, max_tokens)` with `.call(system, user, *, temperature)` returning the raw assistant text. System prompt sent with `cache_control={"type": "ephemeral"}`. API key read from `ANTHROPIC_API_KEY`, falling back to `~/.hermes/.env` parsing. No retries inside (let upstream handle).
  - Verify: `pytest research_loop/tests/test_agent_client.py -q` — mock `anthropic.Anthropic.messages.create`; assert request shape (system block has cache_control; user content correct; no chat history persisted across calls).
  - Files: `research_loop/agents/__init__.py`, `research_loop/agents/client.py`, `research_loop/tests/test_agent_client.py`, `pyproject.toml` (add `anthropic` dep).
  - Commit message: `feat(agents): Anthropic SDK wrapper with prompt caching`

- [ ] **T5. CriticAgent + AuthorBAgent + SynthesizerAgent**
  - Acceptance: three agents in `research_loop/agents/{critic,author,synthesizer}.py` each producing a frozen dataclass result (`Critique`, `PatchProposal`, `Synthesis`). System prompts explicitly:
    - Critic: "find problems only, no fixes proposed"
    - Author B: "produce a unified diff addressing each criticism, no edits outside identified problems, no V1 file edits"
    - Synthesizer: "given two patches A and B with anonymized labels, produce a synthesis that takes the strongest elements of each"
  - User-message input contracts (typed):
    - Critic: trainer source + last 30 rows of `results_v2.tsv` + last 10 Outcomes
    - Author B: critique text + trainer source
    - Synthesizer: just A and B's patches + the original critique (no metric history)
  - Each result preserves `raw: str` for audit.
  - Verify: `pytest research_loop/tests/test_agents.py -q` — `monkeypatch` `AgentClient.call` with canned responses (Critique with 3 problems, AuthorB with a valid unified diff that `git apply --check` accepts on a fixture trainer file, Synthesizer with another valid diff). Assert parsed dataclasses populated; assert each agent uses a fresh user-message (no history mutation).
  - Files: `research_loop/agents/critic.py`, `research_loop/agents/author.py`, `research_loop/agents/synthesizer.py`, `research_loop/tests/test_agents.py`.
  - Commit message: `feat(agents): Critic / Author B / Synthesizer with typed dataclass results`

### ✅ Checkpoint B — Agents complete

```
pytest research_loop/tests/test_agent_client.py \
       research_loop/tests/test_agents.py -q
```

Full suite still green (existing + new agent tests).

---

## Phase 3 — Orchestration

- [ ] **T6. `tournament autoreason` subcommand + convergence loop**
  - Acceptance: new CLI subcommand:
    ```
    tournament autoreason --target {student_v2|dino_v2} \\
        --max-passes 15 --convergence 2 \\
        --max-seconds-per-candidate N \\
        [--hypothesis-seed "..."] [--dry-run]
    ```
    Each pass: refuse if `git status` dirty → Critic → Author B → Synthesizer → for each kind ∈ {A, B, AB}: apply patch (A is no-op), `adapter.train(max_seconds)`, revert → `promote.decide()` → write `Decision`. Track consecutive A wins; exit at `k` or `max_passes`. Exit code 0 on convergence, non-zero on max-passes-exhausted. Persists *every* LLM call's raw text into `history.jsonl` as `record_type ∈ {"critique", "patch_proposal", "synthesis"}`.
  - Verify: `pytest research_loop/tests/test_autoreason_loop.py -q` with mocked LLM + stub adapter:
    - convergence at k=2 (A wins twice) → exit code 0
    - max-passes ceiling → exit code != 0
    - timeout candidate auto-rejected by promote.decide
    - working tree clean after every pass (assert via fixture git repo)
    - history.jsonl contains expected record_type sequence
  - Files: `research_loop/tournament.py` (extend), `research_loop/candidate.py` (add `Critique`/`PatchProposal`/`Synthesis` record types in the discriminated union), `research_loop/tests/test_autoreason_loop.py`.
  - Commit message: `feat(research_loop): autoreason convergence loop + tournament autoreason CLI`

### ✅ Checkpoint C — Orchestration complete

```
.venv/bin/python -m pytest research_loop/tests dino_finetune/tests \
    student_finetune/tests/test_*productness*.py \
    student_finetune/tests/test_adapter_sha.py \
    student_finetune/tests/test_metrics_progress_write.py -q
```

Total tests: 103 (PR #8) + new T1–T6 tests. All green. End-to-end CPU dry-run of `tournament autoreason --max-passes 1 --dry-run` produces a clean run with mocked LLM and mocked subprocess.

---

## Phase 4 — Polish

- [ ] **T7. Audit trail + HANDOFF.md update**
  - Acceptance: every LLM call's raw text appears in `history.jsonl` (not just parsed objects); HANDOFF.md replaces the "Stage 1 / Stage 2 via `run-round --baseline-only`" section with the autonomous-loop launcher; mentions cost ceiling + max-passes + how to interpret convergence vs max-passes-exhausted.
  - Verify: unit test reads a fixture `history.jsonl` from a 3-pass synthetic run and asserts every expected `record_type` occurs (`candidate × 3 kinds × 3 passes` + outcomes + decisions + critique/patch_proposal/synthesis).
  - Files: `research_loop/tournament.py` (audit hooks if not already present), `HANDOFF.md`.
  - Commit message: `docs(handoff): replace run-round with autonomous tournament autoreason flow`

- [ ] **T8. Real-LLM smoke (manual, not CI)**
  - Acceptance: one manual run: `tournament autoreason --target student_v2 --max-passes 1 --dry-run-train` (skip GPU). Real Sonnet 4.6 call. Output: a critique that's plausibly relevant + an Author-B diff that passes `git apply --check` + a Synthesizer diff that also passes. Snippet recorded in HANDOFF.md as one-time evidence.
  - Verify: manual; the snippet IS the verification.
  - Files: `HANDOFF.md`.
  - Commit message: `docs(handoff): real-LLM smoke recorded — autoreason produces applyable diffs`

### ✅ Checkpoint D — Complete

- [ ] All 8 tasks committed atomically on `feat/autoreason-llm-loop`
- [ ] Single PR opened against `master`, body references issue #11
- [ ] Working tree clean
- [ ] Manual T8 evidence recorded
- [ ] Issue #10 stays closed (its scope is fulfilled by T2)
- [ ] Issue #11 closes when this PR merges
