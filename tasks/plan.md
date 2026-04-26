# Implementation Plan: `autoreason --resume` (issue #14)

## Overview

Add candidate-level resume to `tournament autoreason`. After a crash, Ctrl+C,
or machine reboot, the operator runs `autoreason --resume <run_id>` and the
loop picks up from the **last unfinished candidate**, not from the start.
Pass-level resume is implicitly subsumed: if a pass crashed before any
candidate started training, resume just begins a fresh pass.

The key new piece is a `record_type="outcome_started"` record written when a
candidate begins training and matched against the existing `outcome` record
to detect unfinished candidates. Working-tree recovery, PID liveness, and
config restoration round it out.

## Architecture Decisions

1. **`outcome_started` is the state machine.** Three states for a candidate:
   `queued` (Candidate written, no outcome_started) → `running` (outcome_started
   written) → `done` (outcome written, with status=success/failed/timeout).
   "Unfinished" = `outcome_started` present, matching `outcome` absent.

2. **Mid-LLM crash → re-do the whole pass.** Building partial-LLM-state recovery
   (e.g. "critic done, author crashed → only re-run author") triples the
   complexity for marginal LLM cost savings. Cheaper to scan: if the latest
   pass has zero candidates written, just start that pass over.

3. **Working-tree recovery is best-effort revert.** On resume, for each
   unfinished candidate's patch, attempt `git apply -R --check` then
   `git apply -R`. If that fails because the patch wasn't actually applied
   (mid-apply crash, kind="A" no-op), swallow the error — clean tree was
   the goal regardless.

4. **Refuse resume on diverged tree.** If `git status --porcelain` shows
   modifications NOT explained by an unfinished candidate's patch (operator
   committed something, hand-edited a file, pulled new commits), bail with
   a clear error message. Do not silently overwrite.

5. **Resume reuses the original run's LLM config.** Store `llm_cli`,
   `llm_model`, `llm_provider`, per-role overrides in `summary.json` on
   first write; resume reads from there. Operator does not re-pass them.
   Rationale: prevents accidental config drift between pre-crash and
   post-crash passes which would muddy the audit trail.

6. **PID liveness check is mandatory.** If the prior run's PID file exists
   AND the process is alive, refuse resume — there is already a runner.
   Stale PID file (process gone) is fine.

7. **No new run_id on resume.** The run continues under its original
   `<run_id>`; `summary.json` and `autoreason.log` get appended to. The
   `<target>_CURRENT.txt` pointer is updated to match. This keeps the
   audit trail coherent — one run = one run_id.

8. **Pass numbering preserved.** `--max-passes` counts from pass 1 (the
   original start), not from the resume point. If you crashed at pass 5/15
   and resume, the loop runs passes 5–15 (re-running pass 5 if it had no
   completed candidates, else continuing within pass 5).

## Dependency Graph

```
outcome_started record (T1)
    │
    ├── PID + summary + run_dir lookup primitives (T2)
    │       │
    │       └── Working-tree recovery (T3)
    │               │
    │               └── cmd_autoreason --resume orchestrator (T4)
    │                       │
    │                       └── docs + CLI help (T5)
    │
    └── (T2/T3/T4 all consume outcome_started records)
```

Bottom-up: schema first; primitives independent of each other but all
needed by the orchestrator; orchestrator ties them together; doc last.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Working-tree recovery silently corrupts uncommitted work | High | Refuse resume if dirty tree contains changes outside the unfinished candidates' diff. Bail with clear message. |
| `outcome_started` written but Candidate not written → orphan record | Low | Both writes happen inside cmd_autoreason; Candidate is written first. Test the order. |
| Resume on a different LLM CLI than original (operator confusion) | Med | Config stored in summary.json; resume restores it. Operator's `--llm-cli` flag on resume is rejected with explanation. |
| Backward compatibility: old history.jsonl has no outcome_started | Med | `find_unfinished_candidates` returns [] if no outcome_started records exist for the run_id → resume starts a fresh pass safely. Test explicitly. |
| Resume races with a still-alive crashed run (zombie PID file) | Med | PID liveness check via `kill -0`. Refuse if alive. Test both paths. |
| `git apply -R` fails because the patch was never applied | Low | Best-effort: try `--check` first; swallow error if check fails. Tree is verified clean afterward. |
| Resume invocation count: `summary.json` `current_pass` could go backwards | Low | Recompute from history.jsonl on resume; summary is derived state. |
| Promote-without-decision case: outcomes written, decision missing | Med | Detect: if latest round has all candidates with `outcome` but no `decision` → run promote before continuing. |
| Multiple `outcome_started` records for same candidate (resume + recrash) | Med | `find_unfinished` matches the *latest* outcome_started against an outcome by candidate_id; duplicates are tolerated. Test. |

## Out of Scope (deferred)

- Cross-machine resume (run dir + history.jsonl must be on same host)
- Step-level resume inside trainer (already handled by `metrics_progress_v2.json`)
- Resume of `run-round` (only autoreason; run-round is short enough that
  manual restart is fine)
- Mid-pass LLM partial recovery (per ADR #2)

## Task List

### Phase 1: Foundation

- [ ] **T1** — `OutcomeStartedRecord` schema + writer integration

  Add `record_type="outcome_started"` to the `RecordType` Literal. New
  `@dataclass OutcomeStartedRecord` with fields: `candidate_id`, `round_id`,
  `target`, `pass_index`, `started_at`, `kind`, `record_type`. Add
  `read_outcomes_started()` reader and `find_unfinished_candidates(run_id)`
  helper that returns Candidates whose latest outcome_started has no
  matching outcome.

  In `cmd_autoreason`'s candidate loop (line ~672), append
  `OutcomeStartedRecord` *immediately before* `adapter.train()` and after
  `apply_patch`. The Candidate is already in history at this point.

  - **Acceptance:**
    - Schema: `to_jsonl()` / `from_jsonl()` round-trip
    - `find_unfinished_candidates`: started + outcome → finished; started + no
      outcome → unfinished; no started → empty
    - cmd_autoreason writes outcome_started + outcome on success path
    - cmd_autoreason writes outcome_started + no outcome on simulated crash
    - Backward compat: history with no outcome_started → no false positives
    - Existing 154 autoreason tests still pass
  - **Verify:** `pytest research_loop/tests/ -q` shows ≥6 new test cases
  - **Files:** `research_loop/candidate.py`, `research_loop/tournament.py`,
    `research_loop/tests/test_outcome_started.py` (new),
    `research_loop/tests/test_autoreason_loop.py` (extend)
  - **Scope:** S

### Checkpoint: After T1

- [ ] State machine in place: every running candidate is detectable from history alone

### Phase 2: Resume Primitives

- [ ] **T2** — Run-state recovery primitives

  Read-side helpers in a new `research_loop/resume.py` module:
  - `find_run_dir(run_id) -> Path | None` — RUNS_DIR/<run_id>; None if missing
  - `check_pid_dead(run_dir) -> bool` — read pid file; return True if process gone
    or pid file missing; False if alive (uses `os.kill(pid, 0)` probe)
  - `load_run_config(run_dir) -> dict` — read summary.json, extract llm_cli /
    llm_model / llm_provider / per-role overrides / target / max_passes /
    convergence / max_seconds_per_candidate / hypothesis_seed
  - `compute_consecutive_a_wins(target, convergence) -> int` — scan all
    decisions for target, count trailing A wins (capped at convergence)

  Extend `_write_run_summary` to also persist the run's full config block
  on first write (idempotent — on resume, config is read, not re-written).

  - **Acceptance:**
    - All 4 helpers have unit tests against tmp dirs
    - `check_pid_dead`: alive process → False; killed process → True; missing
      pid file → True
    - `load_run_config`: round-trips with `_write_run_summary`
    - `compute_consecutive_a_wins`: 0 for fresh history; resets to 0 on B/AB
      win; caps at convergence
    - summary.json schema extension is backward compatible (old summaries
      without config block load with defaults)
  - **Verify:** new `test_resume_primitives.py` (~8 cases)
  - **Files:** `research_loop/resume.py` (new),
    `research_loop/tournament.py` (`_write_run_summary` extension),
    `research_loop/tests/test_resume_primitives.py` (new)
  - **Scope:** S/M

- [ ] **T3** — Working-tree recovery

  In `research_loop/resume.py`:
  - `recover_working_tree(repo: Path, unfinished: list[Candidate]) -> None`
    - For each unfinished candidate with non-empty patch: try
      `git apply -R --check`. If passes, run `git apply -R`. If fails
      (patch wasn't applied), swallow.
    - After all attempts: `git status --porcelain` must be empty. If not,
      raise `WorkingTreeDivergedError` with the dirty paths listed.
  - Refuse the resume if the dirty paths include any file NOT mentioned
    in the unfinished candidates' diffs (operator hand-edited or pulled).

  - **Acceptance:**
    - Test fixture: tmp git repo with applied patch + 1 unfinished
      candidate → recover succeeds, tree clean
    - Multiple applied patches → all reverted
    - Patch never applied (apply -R --check fails) → silent skip, tree clean
    - Diverged tree (file modified outside any unfinished candidate's diff)
      → raises with clear message
    - kind="A" candidate (empty patch) → no-op
  - **Verify:** new `test_working_tree_recovery.py` (~5 cases)
  - **Files:** `research_loop/resume.py` (extension),
    `research_loop/tests/test_working_tree_recovery.py` (new)
  - **Scope:** S

### Checkpoint: After T3

- [ ] All resume primitives unit-tested in isolation; ready to compose

### Phase 3: Orchestrator

- [ ] **T4** — `cmd_autoreason --resume <run_id>` + crash-matrix tests

  argparse: add `--resume RUN_ID` to `p_auto`. Mutually exclusive with
  the existing run-config flags (operator cannot override LLM config on
  resume; reject with `argparse` error message).

  In `cmd_autoreason`: when `resume` is set, branch into the resume path:
  1. `find_run_dir(run_id)` — error if missing
  2. `check_pid_dead(run_dir)` — error if alive
  3. `load_run_config(run_dir)` — restore target / max_passes / convergence /
     max_seconds_per_candidate / LLM config
  4. `find_unfinished_candidates(history, run_id)` — list of Candidates needing re-run
  5. `recover_working_tree(REPO_ROOT, unfinished)` — clean tree
  6. `compute_consecutive_a_wins(target, convergence)` — restore convergence counter
  7. Determine resume point:
     - If unfinished: re-run unfinished, then check if their round needs promote
     - Else: identify last completed round; if it lacks a decision, run promote;
       otherwise advance pass_index = last_completed_pass + 1
  8. Continue normal main loop until convergence or max_passes

  Update PID file with new process id. Tee stdout/stderr to existing
  autoreason.log (append mode — already correct).

  Crash-matrix integration tests using the existing stub-adapter / mock-LLM
  pattern from `test_autoreason_loop.py`:
  - **Mid-LLM crash**: critic raises → resume sees no records for that pass
    → starts a fresh pass; the partial pass's prior LLM cost is re-paid
  - **Mid-apply crash**: apply_patch raises → no outcome_started written
    → no unfinished candidate → resume skips the candidate, marks it failed,
    or restarts the pass (decision: restart pass — the mid-apply candidate
    can be re-examined by the LLM with knowledge that it was malformed)
  - **Mid-training crash**: outcome_started written, adapter.train raises
    → resume detects unfinished, recovers tree, re-runs candidate
  - **Mid-finally crash**: outcome written but `git apply -R` failed
    → resume detects dirty tree (with applied patch present), reverts
  - **All-clean crash**: all 3 candidates done, decision missing → resume
    runs promote, advances pass_index
  - **PID still alive**: refuse resume with clear error
  - **No history matches run_id**: refuse resume

  - **Acceptance:**
    - `--resume <run_id>` and `--target` are mutually exclusive
      (target comes from summary.json on resume)
    - `--resume` with `--llm-cli` etc → argparse error
    - All 7 crash-matrix scenarios above pass
    - Backward-compat: original 154 autoreason tests still pass
    - Working tree clean after every test (assert in tearDown)
  - **Verify:** `pytest research_loop/tests/test_autoreason_resume.py -q`
    passes (≥7 cases); full suite still green
  - **Files:** `research_loop/tournament.py` (cmd_autoreason refactor),
    `research_loop/tests/test_autoreason_resume.py` (new)
  - **Scope:** M (largest task; ~200 LOC + tests)

### Checkpoint: After T4

- [ ] End-to-end resume works for all 4 crash points + clean-decision-missing
- [ ] Spec acceptance criteria 1–8 from issue #14 all met

### Phase 4: Polish

- [ ] **T5** — Docs (HANDOFF + CLI help) + smoke instructions

  HANDOFF.md new section: `### 2a-bis. Resuming a crashed autoreason run`
  - When to use, exact command, what gets re-run, what gets skipped
  - PID liveness gotcha (don't resume while another runner is going)
  - Diverged-tree recovery instructions (commit your work or use a fresh branch)

  CLI help: ensure `--resume` documentation is clear about config restoration
  ("LLM config is loaded from the run's summary.json; do not re-pass --llm-* flags").

  README.md "Autoreason" section: 1 line + link to HANDOFF.

  - **Acceptance:**
    - HANDOFF section reads start-to-finish for an operator
    - `tournament autoreason --help` mentions `--resume` and config-restoration
    - README link added
  - **Verify:** spot-check rendered docs; manual CLI help inspection
  - **Files:** `HANDOFF.md`, `README.md`, possibly `research_loop/tournament.py`
    (if argparse help text needs updating)
  - **Scope:** XS

### Checkpoint: After T5

- [ ] All issue #14 acceptance criteria met
- [ ] Working tree clean
- [ ] Ready for PR

## Test Plan Summary

| Test file | Cases | What it covers |
|---|---:|---|
| `test_outcome_started.py` (new) | ~6 | record schema, finder helper, backward-compat empty case |
| `test_autoreason_loop.py` (extend) | +2 | autoreason now writes outcome_started; success + simulated crash |
| `test_resume_primitives.py` (new) | ~8 | find_run_dir, check_pid_dead, load_run_config, compute_consecutive_a_wins |
| `test_working_tree_recovery.py` (new) | ~5 | apply/revert recovery, diverged tree refusal, no-op for empty patch |
| `test_autoreason_resume.py` (new) | ~7 | crash matrix end-to-end |

Total new/changed test cases: ~28. Existing 187 tests must continue passing.

## Verification

Before opening PR:
- [ ] All tests pass: `python -m pytest research_loop/tests dino_finetune/tests student_finetune/tests/test_*productness*.py student_finetune/tests/test_adapter_sha.py -q`
- [ ] `tournament autoreason --help` shows `--resume`
- [ ] Manual smoke: simulate crash via Ctrl+C mid-training in a stub-adapter
      autoreason call; verify resume picks up correctly
- [ ] All 5 commits atomic (one per task), conventional-commit style
- [ ] PR body links issue #14, marks all 8 acceptance criteria

## Open Questions for Operator

1. **Mid-apply crash policy.** When `apply_patch` raises (LLM produced an
   invalid diff), should resume:
   - **(a)** restart the pass (re-run Critic / Author / Synthesizer with
     knowledge that the prior B was malformed)
   - **(b)** mark the candidate as failed and continue the round with A and AB

   Plan default: **(a)** — restart the pass. Catches consistent LLM diff
   bugs, costs one pass of LLM tokens. **Confirm or override.**

2. **Resume preserves run_id?** Plan default: **yes** — same run_id, same
   summary.json, same autoreason.log appended. Alternative is "fork into a
   new run_id with a parent reference" which is cleaner-history but harder
   for operators to track. **Confirm.**

3. **Is partial-pass cost-saving worth complicating?** ADR #2 says no
   (re-do the whole pass on mid-LLM crash). Confirm the LLM cost (~$0.10
   per pass) is acceptable to re-pay on crash. **Confirm.**

→ Confirm or correct these and I'll start T1.
