# Plan: Autoreason + Productness CLS (V2)

Companion to [SPEC.md](SPEC.md). Resolved Q5 = (a): 10% deterministic holdout from `REID_PRODUCTS` + `REID_NEGATIVES`.

## Components

### A. `research_loop/` — Autoreason tournament (issue #4)
A1. **Candidate schema** (`candidate.py`) — dataclass + JSONL serializer. Required fields: `id`, `kind ∈ {A,B,AB}`, `target ∈ {student_v2, dino_v2}`, `hypothesis`, `expected_metric`, `changed_files`, `risks`, `rollback`, `parent_incumbent_id`, `patch` (unified diff or no-op), `evidence_refs` (rows from `results_v2.tsv`).
A2. **Judges** (`judges.py`) — rule-based scorers: `proposal_clarity`, `risk_score`, `prior_evidence_consistency` (cross-checks `expected_metric` vs `results_v2.tsv` history). LLM-judge interface stubbed as a plug-in point.
A3. **Evaluators** (`evaluators.py`) — parse `metrics.json` / `run.log` produced by trainers; emit objective scores: `combined`, `recall@1`, `recall@5`, `mean_cosine`, plus productness metrics when present.
A4. **Promote guardrails** (`promote.py`) — noise-band cutoff, recall regression veto, missing-rollback veto, do-nothing-A wins on tie. Pure functions, fully unit-tested.
A5. **Target adapters** (`targets/student_v2.py`, `targets/dino_v2.py`) — uniform interface: `apply_patch(candidate)`, `train()`, `eval()`, `log_row()`. Adapters explicitly assert write paths are V2 only (refuse V1).
A6. **CLI** (`tournament.py`) — `propose | rank | run | promote`. Writes `history.jsonl` and tournament-level `tournament_results_v2.tsv`.
A7. **Protocol doc** (`autoreason_program_v2.md`) — defines do-nothing-A rule, judge rubric, promotion guardrails.

### B. Productness CLS branch (issue #5)
B1. **Val holdout** — deterministic 10% split of `REID_PRODUCTS` + `REID_NEGATIVES` paths (sorted, hashed by SHA-1, modulo bucket). Frozen list materialized once into `student_finetune/productness_val_paths.txt`. Train datasets must exclude these.
B2. **Dataset wrapper** (`ProductnessWrapper` in `train_v2.py`) — adds `is_product ∈ {0.0, 1.0}` per sample by membership in the negatives root.
B3. **Collate** — extend `collate_distill` locally in `train_v2.py` (or use a `collate_distill_v2` wrapper) to emit a 4th tensor `productness_targets`.
B4. **Model** (`ProductnessLCNet` in `train_v2.py`) — subclass `LCNet`, add `productness_head`, expose `forward_train` returning `(embedding, productness_logit)`. `encode()` unchanged.
B5. **Training loop** — extend `run_train_epoch` path with a thin V2 epoch runner that computes `bce_logits(productness_logit, targets)` and adds `cls_w * cls_loss` to the existing total. Logged separately.
B6. **Optimizer** — include `productness_head.parameters()` in optimizer + rebuild paths.
B7. **Validation** — add `eval_productness(model, val_loader) → {loss, acc, pos_acc, neg_acc}`. Logged as separate keys; not folded into `combined`.
B8. **ONNX export** — `export_onnx.py` gains `--include-productness`. Default path remains 1-output, byte-equivalent to V1 within `1e-5`. Productness mode adds a sigmoid output node and tests parity vs PyTorch.
B9. **Feature flags** at top of `train_v2.py`:
```
USE_PRODUCTNESS_CLS = False     # off by default — V2 baseline parity
PRODUCTNESS_CLS_WEIGHT = 0.02
PRODUCTNESS_HEAD_HIDDEN = 256
PRODUCTNESS_VAL_HOLDOUT_FRAC = 0.10
PRODUCTNESS_VAL_SEED = 42
```

## Dependency graph

```
B1 (val holdout list)            ──┐
B2 wrapper ── B3 collate ─┐        │
B4 model ─────────────────┤        ▼
B5 epoch ─────────────────┼──► B7 productness eval ──► B8 ONNX export
B6 optimizer ─────────────┘        ▲
                                   │
A1 schema ── A2 judges ── A4 promote ── A5 adapters ── A6 CLI ── A7 protocol doc
A3 evaluators ───────────────────────────────────────┘
                                                     │
                              productness metrics ───┘ (A3 reads B7's metrics.json keys)
```

A and B can proceed largely in parallel. Only coupling: A3 evaluators must understand productness metric keys (define keys in B5 first).

## Implementation order

**Wave 1 (foundations, parallel):**
- B1: Generate `productness_val_paths.txt` (deterministic, committed).
- A1: `candidate.py` schema + JSONL.
- A4: `promote.py` pure guardrail functions.

**Wave 2 (productness vertical, sequential):**
- B2 → B3 → B4 → B5 → B6
- B7 (depends on B1+B4)
- B8 (depends on B4+B7)

**Wave 3 (tournament vertical, sequential):**
- A2 judges
- A3 evaluators (must include productness keys → schedule after B5 declares them)
- A5 student_v2 adapter
- A5 dino_v2 adapter
- A6 CLI
- A7 protocol doc

**Wave 4 (integration):**
- Unit tests across both verticals (test files declared in SPEC §Testing strategy).
- Smoke run: `python -m research_loop.tournament propose --target student_v2` → produces 3 candidates, judges rank, promote refuses do-nothing tie-break, smoke run on a tiny epoch budget appends row to `results_v2.tsv`.
- V1 parity check: short proxy run of `train.py` / `train_final.py` reproduces V1 metrics within noise.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Productness branch silently regresses retrieval `combined` | Hard success criterion #1 (`combined ≥ 0.8588` with flag off) gated in PR; CI runs flag-off smoke. |
| ONNX default output drifts from V1 | Test `test_export_onnx_productness.py` compares default-mode embedding to a stored V1 reference within `1e-5`. |
| Tournament accidentally edits V1 files | Adapter `__init__` asserts target paths are V2; unit test `test_target_adapters.py` proves V1 writes raise. |
| `REID_NEGATIVES` contains label noise (issue #5 risk) | Phase A logs only; productness ONNX gated behind Phase B val-set numbers. |
| Val holdout leakage if `productness_val_paths.txt` regenerated nondeterministically | Path file committed; generator is pure-function of sorted dataset paths + seed; unit test asserts stability. |
| LLM-judge dependency drift | Out of scope this round — rule-based judges only. |

## Verification checkpoints

- **C1 (after Wave 1):** `pytest research_loop/tests/test_candidate_schema.py research_loop/tests/test_promote_guardrails.py` green; `productness_val_paths.txt` checked in and stable across two regenerations.
- **C2 (after Wave 2):** `pytest student_finetune/tests -q` green; smoke train of `train_v2.py --max-epochs 1 USE_PRODUCTNESS_CLS=False` on 100 samples completes; same with `True` completes and logs productness keys.
- **C3 (after Wave 3):** `python -m research_loop.tournament propose --target student_v2` writes 3 candidates; `rank` selects a winner; `run` on a 1-epoch proxy appends a `results_v2.tsv` row.
- **C4 (final):** Full V2 run with productness ON (the default) achieves `combined ≥ 0.8588` and reports productness pos_acc / neg_acc on the val holdout. ONNX `--include-productness` (the production export path) passes parity tests; legacy embedding-only export still works for V1 compatibility.

> Project decision 2026-04-25: productness is mandatory for V2+ deployment. The flag-off baseline run is no longer a target; `USE_PRODUCTNESS_CLS=True` is the default.
