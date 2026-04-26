# TODO: autoreason --resume (issue #14)

Branch: `feat/autoreason-resume`

## Tasks

- [ ] **T1** `OutcomeStartedRecord` schema + reader + writer in `cmd_autoreason` — new `test_outcome_started.py` (~6 cases) + `test_autoreason_loop.py` extension (+2)
- [ ] **CHECKPOINT** state machine in place — every running candidate detectable from history alone
- [ ] **T2** Run-state recovery primitives (`research_loop/resume.py`): `find_run_dir`, `check_pid_dead`, `load_run_config`, `compute_consecutive_a_wins`; extend `_write_run_summary` with config block — new `test_resume_primitives.py` (~8 cases)
- [ ] **T3** Working-tree recovery: `recover_working_tree`, `WorkingTreeDivergedError` — new `test_working_tree_recovery.py` (~5 cases)
- [ ] **CHECKPOINT** all primitives unit-tested in isolation
- [ ] **T4** `cmd_autoreason --resume <run_id>` orchestrator + 7-scenario crash-matrix — new `test_autoreason_resume.py` (~7 cases)
- [ ] **CHECKPOINT** all 8 issue #14 acceptance criteria met
- [ ] **T5** HANDOFF section §2a-bis + CLI help + README link

## Verification gate before PR

- [ ] `pytest research_loop/tests dino_finetune/tests student_finetune/tests/test_*productness*.py student_finetune/tests/test_adapter_sha.py -q` — all green (≥215 tests)
- [ ] `python -m research_loop.tournament autoreason --help` shows `--resume`
- [ ] One atomic commit per task (5 commits), conventional-commit style
- [ ] Working tree clean after test runs

## Open questions for operator (await confirmation before T1)

1. Mid-apply crash policy: **(a)** restart the pass, or **(b)** mark candidate failed + continue? Default: **(a)**.
2. Resume preserves original run_id (single run = single audit trail)? Default: **yes**.
3. Confirm re-paying ~$0.10 LLM cost on mid-LLM crash is acceptable (vs building partial-LLM recovery)? Default: **acceptable**.
