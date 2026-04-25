# Tasks: Autoreason + Productness CLS (V2)

Companion to [SPEC.md](SPEC.md) and [PLAN.md](PLAN.md). Tasks are ordered by dependency. Each task has acceptance and a verification step. No task should touch more than ~5 files.

---

## Wave 1 — Foundations (parallel-safe)

- [ ] **T1. Generate productness val holdout list**
  - Acceptance: `student_finetune/productness_val_paths.txt` exists with ~10% of (`REID_PRODUCTS` ∪ `REID_NEGATIVES`) paths; deterministic across reruns.
  - Verify: run generator twice, `diff` is empty; `wc -l` is roughly `0.10 * total`.
  - Files: `student_finetune/tools/build_productness_val.py` (new), `student_finetune/productness_val_paths.txt` (new).

- [ ] **T2. Candidate schema (`research_loop/candidate.py`)**
  - Acceptance: `Candidate` dataclass with all fields from PLAN §A1; `to_jsonl` / `from_jsonl` roundtrip.
  - Verify: `pytest research_loop/tests/test_candidate_schema.py -q` passes.
  - Files: `research_loop/__init__.py`, `research_loop/candidate.py`, `research_loop/tests/test_candidate_schema.py`.

- [ ] **T3. Promote guardrails (`research_loop/promote.py`)**
  - Acceptance: pure functions `is_noise_band`, `regresses_recall`, `tie_break_to_incumbent`, `decide(candidate_results) → Decision`.
  - Verify: `pytest research_loop/tests/test_promote_guardrails.py -q` passes (covers small-delta reject, recall-regression veto, tie-break to A).
  - Files: `research_loop/promote.py`, `research_loop/tests/test_promote_guardrails.py`.

## Wave 2 — Productness vertical (sequential)

- [ ] **T4. Productness wrapper + collate**
  - Acceptance: `ProductnessWrapper` in `train_v2.py`; collate emits 4-tuple including `productness_targets` float tensor; `is_product=0` for paths under `REID_NEGATIVES`, `1` otherwise.
  - Verify: `pytest student_finetune/tests/test_productness_targets.py -q` passes.
  - Files: `student_finetune/train_v2.py`, `student_finetune/tests/test_productness_targets.py`.

- [ ] **T5. ProductnessLCNet model**
  - Acceptance: `ProductnessLCNet(LCNet)` adds `productness_head`; `forward_train(images) → (embedding [B,D], logit [B])`; `encode()` unchanged and embedding-only / L2-normalized.
  - Verify: `pytest student_finetune/tests/test_productness_head.py -q` passes.
  - Files: `student_finetune/train_v2.py`, `student_finetune/tests/test_productness_head.py`.

- [ ] **T6. V2 epoch with BCE + optimizer wiring**
  - Acceptance: when `USE_PRODUCTNESS_CLS=True`, training loop computes `bce_with_logits(logit, targets) * PRODUCTNESS_CLS_WEIGHT` and adds to total; head params included in optimizer; per-step / per-epoch logs include `productness_loss`, `productness_acc`.
  - Verify: 1-epoch smoke run on a 100-sample subset prints both keys; total loss > distill loss only when flag is on.
  - Files: `student_finetune/train_v2.py`.

- [ ] **T7. Productness validation eval**
  - Acceptance: `eval_productness(model, val_loader)` returns `{loss, acc, pos_acc, neg_acc}`; called once per epoch when flag on; results NOT folded into `combined`.
  - Verify: smoke run logs `productness_acc`, `productness_pos_acc`, `productness_neg_acc` ≠ NaN; retrieval `combined` unchanged numerically when flag off.
  - Files: `student_finetune/train_v2.py`.

- [x] **T8. Feature flags (productness ON by default per project decision 2026-04-25)**
  - Acceptance: `USE_PRODUCTNESS_CLS = True` is the default; the off-path remains for debugging only and is covered by `test_v1_default_path_unchanged`.
  - Verify: V1 unit-test pass count unchanged after additive train.py edits.
  - Files: `student_finetune/train_v2.py`.

- [ ] **T9. ONNX export — default V1-compatible + `--include-productness`**
  - Acceptance: `python export_onnx.py` produces 1-output ONNX whose embedding matches a stored V1 reference within `1e-5` on a fixed input. `python export_onnx.py --include-productness` produces 2-output ONNX whose `productness_score ∈ [0,1]` matches PyTorch sigmoid within `1e-5`.
  - Verify: `pytest student_finetune/tests/test_export_onnx_productness.py -q` passes (CPU-only, uses tiny dummy weights).
  - Files: `student_finetune/export_onnx.py`, `student_finetune/tests/test_export_onnx_productness.py`, `student_finetune/tests/fixtures/v1_embedding_reference.npy`.

## Wave 3 — Tournament vertical (sequential)

- [ ] **T10. Rule-based judges (`research_loop/judges.py`)**
  - Acceptance: `score_clarity`, `score_risk`, `score_prior_evidence` return floats in [0,1]; `rank(candidates) → ordered list` deterministically ties to incumbent A on ties; LLM-judge interface stubbed (`Judge` protocol).
  - Verify: unit test in `research_loop/tests/test_judges.py` covers ordering, tie-break, missing-evidence penalty.
  - Files: `research_loop/judges.py`, `research_loop/tests/test_judges.py`.

- [ ] **T11. Evaluators (`research_loop/evaluators.py`)**
  - Acceptance: parses `metrics.json` and `run.log` produced by `train_v2.py` and `train_dino_v2.py`; emits `combined`, `recall@1`, `recall@5`, `mean_cosine`, plus productness keys when present.
  - Verify: `pytest research_loop/tests/test_evaluators.py -q` against fixture log files.
  - Files: `research_loop/evaluators.py`, `research_loop/tests/test_evaluators.py`, `research_loop/tests/fixtures/sample_metrics.json`.

- [ ] **T12. Target adapter — student_v2**
  - Acceptance: `targets/student_v2.py` exposes `apply_patch`, `train`, `eval`, `log_row`; asserts log path is `student_finetune/results_v2.tsv` (refuses V1); refuses to apply a patch touching V1 files.
  - Verify: `pytest research_loop/tests/test_target_adapters.py::test_student_v2 -q`.
  - Files: `research_loop/targets/__init__.py`, `research_loop/targets/student_v2.py`, `research_loop/tests/test_target_adapters.py`.

- [ ] **T13. Target adapter — dino_v2**
  - Acceptance: same interface, writes only to `dino_finetune/results_v2.tsv`; refuses V1 paths.
  - Verify: `pytest research_loop/tests/test_target_adapters.py::test_dino_v2 -q`.
  - Files: `research_loop/targets/dino_v2.py`, `research_loop/tests/test_target_adapters.py`.

- [ ] **T14. Tournament CLI (`research_loop/tournament.py`)**
  - Acceptance: subcommands `propose | rank | run | promote`; `propose` always emits an A-incumbent candidate; `run` is target-agnostic via adapter; appends to `research_loop/history.jsonl` and the appropriate `results_v2.tsv`.
  - Verify: `python -m research_loop.tournament propose --target student_v2` produces ≥3 candidates; `rank` writes ordering; `run --candidate <A_id>` is a no-op with logged "do-nothing" outcome.
  - Files: `research_loop/tournament.py`, `research_loop/history.jsonl` (created on first run).

- [ ] **T15. Tournament protocol doc**
  - Acceptance: `research_loop/autoreason_program_v2.md` documents do-nothing-A rule, candidate schema, rule-based judge rubric, promotion guardrails, V2-only write rule.
  - Verify: doc references `results_v2.tsv` (not v1); references `train_v2.py` and `train_dino_v2.py`.
  - Files: `research_loop/autoreason_program_v2.md`.

## Wave 4 — Integration & verification

- [ ] **T16. Cross-cutting test run**
  - Acceptance: `python -m pytest student_finetune/tests research_loop/tests -q` is green on CPU.
  - Verify: CI-equivalent local invocation passes.
  - Files: none (test orchestration only).

- [ ] **T17. End-to-end smoke tournament (student_v2)**
  - Acceptance: full `propose → rank → run → promote` cycle on a 1-epoch proxy budget appends one row to `student_finetune/results_v2.tsv` and one entry to `research_loop/history.jsonl`.
  - Verify: row schema matches existing `results_v2.tsv` columns; history entry references the winning candidate id.
  - Files: appends only.

- [ ] **T18. End-to-end smoke tournament (dino_v2)**
  - Acceptance: same as T17 but on `train_dino_v2.py`; appends to `dino_finetune/results_v2.tsv`.
  - Verify: row appended; history entry references DINO target.
  - Files: appends only.

- [ ] **T19. V1 parity sanity check**
  - Acceptance: short proxy run of `train.py` / `train_final.py` / `train_dino.py` with V1 configs produces metrics within noise of pre-existing V1 logs; no V1 file modified by tournament code.
  - Verify: `git status` shows no V1 file modified; metric diff < `1e-3`.
  - Files: read-only verification.

- [ ] **T20. Final V2 production run (productness ON by default)**
  - Acceptance: full V2 run reaches `combined ≥ 0.8588` with productness on; productness `pos_acc`, `neg_acc` reported on the val holdout each epoch.
  - Verify: row appended to `student_finetune/results_v2.tsv` with `description="v2 productness on (production)"`.
  - Files: appends only.
  - Note: per project decision 2026-04-25, no separate "flag off" baseline run is needed.

---

## Out of scope (defer to follow-up issues)

- LLM-based judges (interface stubbed only).
- Hard-negative mining for productness (Phase C in issue #5).
- Tuning `PRODUCTNESS_CLS_WEIGHT` above `0.05` (requires user approval).
- Teacher-aware data curation (Phase 3 in issue #4).
- Research memory / preference data (Phase 4 in issue #4).
