# TODO: `propose --variants` (issue #9 Goal 3)

Branch: `feat/batch-propose-variants`

## Tasks

- [ ] **T1** Variants file schema + loader (`_load_variants`, JSONL parse, validation) — `tournament.py` + new `test_variants.py`
- [ ] **T2** `propose --variants <path>` CLI flag (mutex with `--baseline-only`, emits 1 A + N B) — `tournament.py` + test
- [ ] **CHECKPOINT** all tests pass; manual `propose --variants` sanity check
- [ ] **T3** Verify `decide()` + `rank()` handle N>2 — test-only, no production change
- [ ] **CHECKPOINT** spec assumption verified
- [ ] **T4** `run-round --variants` passthrough — `tournament.py` + test
- [ ] **T5** Productness weight sweep recipe (`sweeps/productness_weight.jsonl`) + HANDOFF doc
- [ ] **CHECKPOINT** acceptance criteria met, working tree clean, ready for PR

## Verification gate before PR

- [ ] `pytest research_loop/tests dino_finetune/tests student_finetune/tests/test_*productness*.py student_finetune/tests/test_adapter_sha.py -q` — all green
- [ ] `python -m research_loop.tournament propose --help` shows `--variants`
- [ ] All 4 productness-sweep diffs pass `git apply --check`
- [ ] One atomic commit per task (5 commits total), conventional-commit style

## Notes / open questions for operator

1. Confirm variants mode does NOT auto-synthesize an AB — assumed default.
2. Confirm convention `research_loop/sweeps/<name>.jsonl` for sweep files.
3. Variants run sequentially on one GPU (not parallelized).
