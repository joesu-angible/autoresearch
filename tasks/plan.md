# Implementation Plan: `propose --variants` (issue #9 Goal 3)

## Overview

Add an N-way candidate sweep mode to the tournament CLI. Operator supplies a
JSONL file describing N candidate patches; `propose --variants <file>` emits
one A baseline + N candidates sharing a single `round_id`. `rank` and `decide`
then work over the full set, with the existing guardrails enforced uniformly.

First use case: sweep `PRODUCTNESS_CLS_WEIGHT` across {0.0, 0.01, 0.05, 0.1}
against the current incumbent (0.02 = A) on `student_v2`.

## Architecture Decisions

1. **Variants reuse `kind="B"`.** `CandidateKind = Literal["A", "B", "AB"]` is
   not extended. Each variant gets a unique `id`; downstream code already
   iterates by id, not by kind. Avoids schema migration of `history.jsonl`
   and `results_v2.tsv`.
2. **No `Round` container dataclass.** A round is N candidates sharing a
   `round_id` string — already the case. No new abstraction.
3. **Variants mode skips AB synthesis.** AB is "halve B's magnitude" — meaningless
   for a fixed sweep. `--variants` is mutually exclusive with `--baseline-only`
   and replaces both placeholder B and placeholder AB emission.
4. **JSONL schema = subset of Candidate.** One JSON object per line with the
   required Candidate fields except `kind`, `id`, `round_id`, `target` (filled
   by `cmd_propose`). Forces the operator to specify `hypothesis`,
   `expected_metric`, `changed_files`, `risks`, `rollback`, `patch`.
5. **Patch validation deferred to `cmd_run`.** `propose` only checks JSONL
   parses and required fields exist. The first call to `apply_patch` in `cmd_run`
   does the real `git apply --check`. Reason: keeps `propose` cheap and offline.
6. **Issue #9 Goals 1 + 2 stay separate.** Goal 1 (time budget) is already done.
   Goal 2 (proxy → full multi-stage) is NOT in scope here.

## Dependency Graph

```
Variants file schema (Task 1)
    │
    └── cmd_propose --variants flag (Task 2)
            │
            ├── cmd_rank — verify N>2 ordering (Task 3, no code change)
            ├── promote.decide — verify N>2 picks best (Task 3, no code change)
            │
            └── cmd_run_round --variants passthrough (Task 4)
                    │
                    └── productness weight sweep recipe (Task 5)
```

Bottom-up: schema first (Task 1), then producer (Task 2), then verify consumers
already work (Task 3), then orchestrator (Task 4), then real-world payload (Task 5).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Variants file with malformed patch reaches `cmd_run` | Med | `apply_patch` already raises on `git apply --check` failure → outcome `status="failed"` → `decide()` skips it. Same as today. |
| Operator forgets `parent_incumbent_id` on variants | Low | `cmd_propose` fills it from the auto-emitted A's id. Operator never sets it. |
| 5+ variants × 9000s budget = 12.5h round | Med | Document in HANDOFF: variants mode lengthens rounds proportionally. Operator picks N consciously. |
| `--variants` + `--baseline-only` both set | Low | argparse mutually exclusive group — caught at parse time. |
| Existing tests assume A/B/AB triple shape | Med | Inspect `test_tournament.py` and `test_autoreason_loop.py`; verify they assert on kind set, not count. Likely fine since they iterate. |
| `decide()` claim "handles N>2" untested | High | Task 3 explicitly proves it with a test. |

## Out of Scope (deferred)

- Issue #9 Goal 2 (proxy → full multi-stage rounds)
- Variants integration into autoreason LLM loop (autoreason still emits A/B/AB triple)
- GUI / dashboard for sweep results

## Task List

### Phase 1: Foundation

- [ ] **Task 1** — Variants file schema + loader
  - Define `_load_variants(path: Path) → list[VariantSpec]` in `tournament.py`
  - `VariantSpec` is a TypedDict / dataclass with: `hypothesis`, `expected_metric`,
    `changed_files`, `risks`, `rollback`, `patch`
  - Parse JSONL line-by-line, raise `ValueError` with line number on malformed
  - **Acceptance:**
    - Valid 3-line file loads to 3 `VariantSpec` objects
    - Missing required field → `ValueError` mentioning the missing field
    - Malformed JSON → `ValueError` mentioning the line number
    - Empty file → empty list (caller decides if that's an error)
  - **Verify:** `pytest research_loop/tests/test_variants.py -q` passes (≥4 cases)
  - **Files:** `research_loop/tournament.py`, `research_loop/tests/test_variants.py`
  - **Scope:** S (1 file edit + 1 new test file)

- [ ] **Task 2** — `propose --variants <path>` CLI flag
  - argparse: add `--variants PATH` to `p_propose`, mutually exclusive with `--baseline-only`
  - In `cmd_propose`: if `variants` set, emit A + one Candidate per variant (kind="B",
    `parent_incumbent_id=a.id`, all with same `round_id`); skip placeholder B/AB
  - Print one line per emitted candidate with id + truncated hypothesis
  - **Acceptance:**
    - `propose --target student_v2 --variants test.jsonl` (3 lines) writes
      1 A + 3 B records to `history.jsonl`, all sharing one `round_id`
    - `--variants` and `--baseline-only` both → argparse error
    - Variants without `parent_incumbent_id` get A's id auto-filled
  - **Verify:** new test in `test_tournament_propose.py` (or appropriate file)
    asserts the 4 records on disk; existing A/B/AB tests still pass
  - **Files:** `research_loop/tournament.py`, `research_loop/tests/test_tournament_propose.py`
  - **Scope:** S (1 file edit + 1 test)

### Checkpoint: After Tasks 1–2

- [ ] All new tests pass
- [ ] `pytest research_loop/tests/ -q` shows no regressions in existing 164 tests
- [ ] Manual: `python -m research_loop.tournament propose --target student_v2 --variants <fixture>` produces sane stdout

### Phase 2: Verify Consumers Handle N>2

- [ ] **Task 3** — Prove `decide()` and `rank()` work for N>2
  - Add test in `test_promote.py`: 1 A + 5 B with varied combined values
  - Assert `decide()` returns the highest-combined B that passes all guardrails
  - Assert that with all 5 B regressed, A wins by default
  - Assert that the highest-combined B with `productness_neg_acc` regression is skipped
    in favor of the second-highest
  - Add similar test to `test_tournament_rank.py` (or wherever `rank_candidates` is tested):
    1 A + 5 B with varied scores → rank order matches sorted score
  - **Acceptance:**
    - 3 new test cases in `decide()` cover: N>2 happy path, all-regress fallback, guardrail veto with N>2
    - 1 new test case in rank covers N>2 ordering
  - **Verify:** `pytest research_loop/tests/test_promote.py research_loop/tests/test_tournament_rank.py -q` (or equivalent paths) passes
  - **Files:** existing test files (no production code changes)
  - **Scope:** XS (test-only)

### Checkpoint: After Task 3

- [ ] Spec assumption verified: decide() and rank() handle N>2 with no code changes

### Phase 3: Orchestrator + Recipe

- [ ] **Task 4** — `run-round --variants <path>` passthrough
  - argparse: add `--variants PATH` to `p_round`
  - `cmd_run_round` forwards to `cmd_propose(..., variants=path)`
  - Verify the existing iteration loop (line ~280) handles N candidates without
    A/B/AB-specific logic; if it does, no change. If it special-cases kinds, fix.
  - **Acceptance:**
    - `run-round --target student_v2 --variants test.jsonl --dry-run` runs
      A + N candidates and exits 0
    - Final promote step writes one Decision over all N+1 candidates
  - **Verify:** new test in `test_autoreason_loop.py` or `test_tournament_run_round.py`
    using mock target with stub outcomes
  - **Files:** `research_loop/tournament.py`, test
  - **Scope:** S

- [ ] **Task 5** — Productness weight sweep recipe + doc
  - Create `research_loop/sweeps/productness_weight.jsonl` with 4 variants:
    weights {0.0, 0.01, 0.05, 0.1} (A is current 0.02). Each entry has a
    real unified diff against `student_finetune/train_v2.py`.
  - Verify each diff with `git apply --check <(jq -r .patch <line>)` (or equivalent)
  - Add a section to `HANDOFF.md` (or new `SWEEPS.md`) documenting how to run
    a sweep, expected duration, decision rule
  - **Acceptance:**
    - All 4 patches pass `git apply --check`
    - Doc shows the exact command an operator runs
    - Decision rule from the metric-strategy memory is reproduced verbatim
  - **Verify:** `propose --variants research_loop/sweeps/productness_weight.jsonl --target student_v2`
    writes 1 A + 4 B candidates to history. (Don't actually run training.)
  - **Files:** new sweep file, `HANDOFF.md` or new `SWEEPS.md`
  - **Scope:** S

### Checkpoint: After Task 5

- [ ] All acceptance criteria met
- [ ] Working tree clean (no leftover sweep diffs applied)
- [ ] Issue #9 Goal 3 fully addressable via this branch
- [ ] Ready for PR

## Test Plan Summary

| Test file | Cases | What it covers |
|---|---:|---|
| `test_variants.py` | ~4 | JSONL parse, missing field, malformed, empty |
| `test_tournament_propose.py` | ~3 | --variants writes A+N, mutex with --baseline-only, parent_id auto-fill |
| `test_promote.py` (extended) | +3 | decide() N>2 happy/regress/veto |
| `test_tournament_rank.py` (extended) | +1 | rank() orders N>2 |
| `test_tournament_run_round.py` | +1 | run-round --variants end-to-end dry-run |

Total new/changed test cases: ~12. Existing 164 tests must continue passing.

## Verification

Before opening PR:
- [ ] All tests pass: `python -m pytest research_loop/tests dino_finetune/tests student_finetune/tests/test_*productness*.py student_finetune/tests/test_adapter_sha.py -q`
- [ ] CLI help shows `--variants` flag
- [ ] Sample sweep file dry-runs cleanly
- [ ] HANDOFF doc updated
- [ ] Single-branch, one commit per Task (5 commits) following repo convention
- [ ] PR body links issue #9, marks Goal 3 closeable

## Open Questions for Operator

1. **Auto-synthesize an AB across variants?** No — sweeps don't synthesize.
   Confirmed by ADR above. Default unless operator says otherwise.
2. **Variants file location convention?** Proposing `research_loop/sweeps/<name>.jsonl`.
   Alternative: caller-relative arbitrary path. Going with the convention by default.
3. **Should `cmd_run_round` parallelize variants?** No — variants run sequentially
   on the single GPU. Documenting expected duration in HANDOFF.
