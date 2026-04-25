# Spec: Autoreason Integration + Productness CLS Branch (V2)

Tracks GitHub issue [#6](https://github.com/joesu-angible/autoresearch/issues/6) (umbrella) and its children #4 (Autoreason tournament) and #5 (productness CLS branch).

**Branch:** `feat/autoreason-cls-umbrella`

## Scope override (V1 → V2)

The umbrella issue and its children reference V1 files (`train.py`, `train_final.py`, `train_dino.py`, `results.tsv`, etc.). Per user direction, **all V1 references in those issues are redirected to V2 equivalents** for this work.

Two distinct rule contexts (clarified by user):

- **Development time** (this branch's PRs): we may edit `prepare.py`, `train.py`, `train_dino.py`, etc. when productness wiring or shared infrastructure genuinely needs it. The hard constraint is that V1 entrypoints (`train.py` / `train_final.py` / `train_dino.py`) must keep producing identical results to V1 when run with V1 configs — V2 changes are additive and gated by feature flags.
- **Tournament time** (autoresearch outer loop running): `prepare.py`, V1 trainers, V1 `results.tsv`, and V1 program docs are read-only. The tournament may only modify V2 trainers and V2 logs.

| File | Dev-time | Tournament-time |
|---|---|---|
| `student_finetune/train_v2.py` | edit freely | tournament patches here |
| `dino_finetune/train_dino_v2.py` | edit freely | tournament patches here |
| `student_finetune/results_v2.tsv` / `dino_finetune/results_v2.tsv` | append | append |
| `student_finetune/prepare.py` | edit if necessary, additive only | read-only |
| `student_finetune/train.py` / `train_final.py` / `dino_finetune/train_dino.py` | edit if necessary, must keep V1 behavior | read-only |
| `student_finetune/results.tsv` / `dino_finetune/results.tsv` (V1 logs) | read-only | read-only |

Baselines to beat: student `combined=0.8588` (V1 best, commit `87766d9`); DINO V2 baseline TBD on first clean run.

## Objective

1. **Autoreason tournament (issue #4):** Add an outer-loop experiment controller around the V2 training pipeline. Each proposed change to `train_v2.py` (student) and `train_dino_v2.py` (DINO) must compete A/B/AB against a do-nothing incumbent, be blind-judged, and only the winning candidate consumes GPU. Promote based on objective retrieval eval, not judge ranking.
2. **Productness CLS branch (issue #5):** Add a binary `1=product / 0=personal-item` head on top of the shared LCNet summary feature, behind a feature flag. Embedding/retrieval path unchanged by default; default ONNX export remains embedding-only / V1-compatible; opt-in `(embedding, productness_score)` export added.

Success = both subsystems land on `feat/autoreason-cls-umbrella` without regressing the V2 student baseline (`combined ≥ 0.8588`) when the productness branch is disabled.

## Tech stack

- Python 3 + PyTorch (existing env), `loguru`, `timm`, `transformers` (DINOv3), ONNX export.
- No new ML framework dependencies. Tournament/judging code is plain Python + JSONL artifacts.
- Existing data roots in `prepare.py`: `TRAIN_DIR`, `REID_PRODUCTS`, `REID_COMMODITY`, `ARCFACE_DIR`, `REID_NEGATIVES`.

## Commands

```
# Student V2 training
cd student_finetune && python train_v2.py --max-epochs 30 > run_v2.log 2>&1

# DINO teacher V2 training
cd dino_finetune && python train_dino_v2.py > run_v2.log 2>&1

# ONNX export (default: embedding-only, V1-compatible)
cd student_finetune && python export_onnx.py

# ONNX export with productness score
cd student_finetune && python export_onnx.py --include-productness

# Tournament: propose A/B/AB candidates for the next train_v2 change
python -m research_loop.tournament propose --target student_v2   # or dino_v2

# Tournament: run winning candidate end-to-end (apply patch → train → eval → log)
python -m research_loop.tournament run --candidate <id>

# Tests
python -m pytest student_finetune/tests research_loop/tests -q
```

## Project structure

User direction: tournament code lives in its own folder for safety / isolation.

```
autoresearch/
├── dino_finetune/
│   ├── train_dino.py               # V1 (read-only at tournament time; V1-behavior-preserving edits OK in dev)
│   ├── train_dino_v2.py            # V2 teacher — tournament target
│   ├── results.tsv                 # V1 log (read-only)
│   ├── results_v2.tsv              # V2 DINO experiment log
│   ├── program_dino.md             # V1 protocol
│   └── program_dino_v2.md          # V2 DINO protocol
├── student_finetune/
│   ├── prepare.py                  # additive edits OK in dev; tournament-time read-only
│   ├── train.py                    # V1 (read-only at tournament time; V1-behavior-preserving edits OK in dev)
│   ├── train_final.py              # V1 (read-only at tournament time)
│   ├── train_v2.py                 # V2 student trainer — productness branch wired here, tournament target
│   ├── export_onnx.py              # extended with --include-productness flag
│   ├── results.tsv                 # V1 log (read-only)
│   ├── results_v2.tsv              # V2 student experiment log
│   ├── program_final_student_v2.md # V2 student protocol
│   └── tests/
├── research_loop/                  # NEW — Autoreason tournament outer loop (isolated folder)
│   ├── __init__.py
│   ├── candidate.py                # Candidate dataclass + JSONL schema
│   ├── tournament.py               # propose / blind-rank / run / promote CLI
│   ├── judges.py                   # rule-based proposal/risk/prior-evidence rubric (LLM plug-in later)
│   ├── evaluators.py               # parse metrics.json / run.log → objective scores
│   ├── promote.py                  # guardrails (noise band, regression, leakage checks)
│   ├── targets/
│   │   ├── student_v2.py           # adapter: how to train+eval+log a student-v2 candidate
│   │   └── dino_v2.py              # adapter: how to train+eval+log a dino-v2 candidate
│   ├── history.jsonl               # append-only tournament history
│   ├── tournament_results_v2.tsv   # tournament-level summary (separate from per-trainer results_v2.tsv)
│   ├── autoreason_program_v2.md    # protocol doc (V2 baselines, do-nothing rules)
│   └── tests/
└── SPEC.md                         # this file
```

## Code style

Follow existing `train_v2.py` style: module-level UPPER_SNAKE config constants, `loguru` for logs, dataclasses for stats. Productness additions live primarily in `train_v2.py`. Edits to `prepare.py` / `train.py` are allowed during development if they keep V1 default behavior identical (add a kwarg defaulted to V1 behavior, do not change existing call sites). Prefer minimal additive changes over rewrites.

```python
# train_v2.py — local extension preferred; subclassing keeps V1 LCNet default-equivalent
class ProductnessLCNet(LCNet):
    """LCNet + auxiliary binary productness head on the shared summary feature."""
    def __init__(self, *a, productness_hidden: int = 256, **kw):
        super().__init__(*a, **kw)
        self.productness_head = nn.Sequential(
            nn.Linear(1280, productness_hidden),
            nn.BatchNorm1d(productness_hidden),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(productness_hidden, 1),
        )

    def forward_train(self, images):
        spatial, summary = self.forward_features(images)
        embedding = functional.normalize(self.proj(summary), dim=-1)
        productness_logit = self.productness_head(summary).squeeze(1)
        return embedding, productness_logit
```

Productness targets: prefer a local `ProductnessWrapper` over `CombinedDistillDataset` in `train_v2.py`. If a small additive `productness_targets=False` kwarg on `prepare.build_distill_dataset` (or similar) is cleaner, that is also acceptable in dev — the only hard rule is V1 default behavior must not change.

```python
# train_v2.py
class ProductnessWrapper(Dataset):
    def __init__(self, base: CombinedDistillDataset, negative_paths: set[str]):
        self.base, self.negative_paths = base, negative_paths
    def __getitem__(self, i):
        img, label, path = self.base[i]
        is_product = 0.0 if path in self.negative_paths else 1.0
        return img, label, path, is_product
```

## Testing strategy

- `student_finetune/tests/` (pytest, existing). Add:
  - `test_productness_targets.py` — wrapper assigns `is_product=0` for `REID_NEGATIVES` paths, `1` otherwise; product/blacklist mix in a sample batch is non-empty under default config.
  - `test_productness_head.py` — `ProductnessLCNet.forward_train` returns `(embedding [B, D], logit [B])`; `encode()` is unchanged and embedding-only.
  - `test_export_onnx_productness.py` — `--include-productness` produces a 2-output ONNX; default produces a 1-output ONNX whose embedding tensor matches the V1 export within `1e-5` for a fixed input.
- `research_loop/tests/`:
  - `test_candidate_schema.py` — JSONL roundtrip; required fields (`hypothesis`, `expected_metric`, `changed_files`, `risks`, `rollback`).
  - `test_promote_guardrails.py` — noise-band rejects small deltas; recall@1 regression vetoes a mean_cosine win; do-nothing baseline A wins on tie.
  - `test_target_adapters.py` — `student_v2` and `dino_v2` adapters expose the required interface (`apply_patch`, `train`, `eval`, `log_row`) and refuse to write into V1 files.
- **No GPU required for unit tests** — gate full-train integration behind `pytest -m gpu`.
- Coverage target: new code ≥ 80% line coverage; no target on `train_v2.py` itself (integration-tested by actual runs into `results_v2.tsv`).

## Boundaries

**Always:**
- Append tournament outcomes to `research_loop/history.jsonl` plus the appropriate `results_v2.tsv` (student or DINO).
- Include a do-nothing incumbent A as a formal candidate in every proposal round.
- Run `python -m pytest student_finetune/tests research_loop/tests -q` before any commit on this branch.
- Keep `encode()` on `LCNet` / `ProductnessLCNet` embedding-only and L2-normalized.
- Default ONNX export = embedding-only (V1-compatible).
- Productness branch must be feature-flagged off by default in V2 trainers; flipping the flag must be the only difference between V1-parity behavior and productness-on behavior.

**Ask first:**
- Adding new dataset roots or changing `CombinedDistillDataset` source-root semantics.
- Tuning `PRODUCTNESS_CLS_WEIGHT` above `0.05` — start at `0.02`.
- Promoting a tournament winner whose `recall@1` regresses, even if `combined` improves.
- Promoting a winner that lacks a robustness/blacklist eval delta.
- Any non-additive edit to `prepare.py` / V1 trainer files (additive, default-off changes do not need approval).

**Never:**
- Write tournament results into V1 `results.tsv` files.
- Replace the objective retrieval eval with LLM-judge scores as the promotion authority.
- Copy code from `NousResearch/autoreason` (license unclear); reimplement the pattern only.
- Ship a productness-enabled ONNX as the default export.
- Use `--no-verify` to bypass tests.
- Have the tournament outer loop auto-patch `prepare.py` or any V1 file.

## Two-sided productness (project decision 2026-04-25)

Productness is added on **both** the teacher (DINOv3 + LoRA) and the student (LCNet). Project rationale: shaping the teacher's 1280-d embedding space to be product-aware before distillation gives the student a head-start on the same signal — productness is mandatory across the stack, not just at the deployment layer.

- `dino_finetune/train_dino_v2.py` — `DinoProductnessHead` on the CLS embedding; target derived from the existing `NEGATIVE_LABEL` sentinel (cleaner than path-string match).
- `student_finetune/train_v2.py` — `ProductnessLCNet` head on the shared summary feature; target derived from `REID_NEGATIVES` membership.
- Identical loss math both sides (`productness_loss_block` in DINO mirrors the inline block in `train.py`): asymmetric label smoothing (`eps_pos=0.05`, `eps_neg=0.02`) + focal weighting (`γ=2.0`).
- Teacher saves `productness_head.pt` alongside the LoRA adapter. Not consumed at student-distillation time (student reads cached embeddings only); its purpose is to shape the embedding space at *teacher* training time.

## Metric model (project decision 2026-04-25)

`combined_metric` stays **retrieval-only**: `0.5 * recall@1 + 0.5 * mean_cosine`. Comparable to V1 history, no re-labeling.

Productness enters the loop as **separate guardrails**, not a weighted blend, in two places:

1. **Tournament promotion** (`research_loop/promote.py::decide`) — adds `productness_neg_acc` regression veto (`> 0.02` drop kills promotion). A challenger that improves `combined` while tanking personal-item rejection cannot win.
2. **Deployment gate** (`research_loop/promote.py::is_deployable`) — absolute thresholds: `combined ≥ 0.86`, `productness_pos_acc ≥ 0.97`, `productness_neg_acc ≥ 0.85`. Independent of tournament rank.

A candidate can be promoted (it's the new best) yet still fail the deployment gate.

## Success criteria

> **Project decision 2026-04-25:** all V2+ student models must ship with productness on. `USE_PRODUCTNESS_CLS` defaults to `True`; criteria below assume the flag is on. The flag-off path remains as a debugging escape hatch and proves the V1 default behavior is preserved, but is not a deployment target.

1. With `USE_PRODUCTNESS_CLS = True`, `train_v2.py` produces a run whose `combined ≥ 0.8588` (productness must not regress retrieval below the V1 best). Productness `pos_acc` and `neg_acc` are logged separately each epoch and reach non-trivial values on the val holdout (smoke run already shows `pos_acc ≈ 0.99`, `neg_acc ≈ 0.76` after 1 epoch).
2. The dataloader path used by `run_train_epoch` derives `is_product` from `REID_NEGATIVES` membership; productness BCE adds `PRODUCTNESS_CLS_WEIGHT * loss` to the joint objective; productness keys appear in `metrics_final_v2.json`.
3. With `USE_PRODUCTNESS_CLS = False` (debug only), `train.py`/`train_final.py` and the V1 unit test set behave identically to the pre-branch state. Verified by `test_v1_default_path_unchanged`.
4. `export_onnx.py --include-productness` produces a 2-output ONNX whose `productness_score ∈ [0, 1]` matches the PyTorch sigmoid within `1e-5` tolerance on a fixed input. This is the production export path going forward.
5. The legacy `export_onnx.py` (embedding-only) still works for V1-checkpoint compatibility but is not the deployment target.
5. `research_loop/` exposes a working CLI that:
   - Generates ≥3 candidates (A=incumbent, B=patch, AB=synthesis) into `research_loop/history.jsonl`.
   - Runs the winning candidate end-to-end on **both** student-v2 and DINO-v2 targets and appends one row to the corresponding `results_v2.tsv`.
   - Refuses to promote on noise-band deltas, recall regressions, or missing rollback condition.
   - Refuses to write into V1 files (verified by `test_target_adapters.py`).
6. All new unit tests pass under `python -m pytest -q` (CPU-only).
7. V1 files unchanged in semantics: running `train.py` / `train_final.py` / `train_dino.py` with their V1 configs reproduces V1 metrics within noise (manual sanity check on a short proxy run).

## Resolved decisions (from user)

- **Tournament-time vs dev-time immutability:** the "do-not-touch V1" rule applies during tournament runs, not during this branch's development. Encoded in the table above and the boundaries section.
- **Tournament code location:** new isolated folder `research_loop/` (user preference for safety).
- **ONNX productness:** must satisfy V2; opt-in via `--include-productness`, default stays V1-compatible.
- **Judges:** I'll start rule-based; LLM judge plug-in deferred.
- **Tournament scope:** covers both student-v2 and DINO-v2 from day one (user: "通通都要").

## Open question (re-asked simpler)

**Q5 (productness validation split):** the productness CLS branch needs its own held-out validation set to prove the head is actually learning product-vs-personal-item — current retrieval eval doesn't measure this. Two ways to build that val set:

- **(a)** Carve out a fixed 10% slice of `REID_PRODUCTS` + `REID_NEGATIVES` (deterministic seed) and exclude those paths from training. This gives us productness accuracy / pos-acc / neg-acc numbers from epoch 1.
- **(b)** Skip the val split for now: only log productness *training* loss/accuracy in Phase A, build a proper val split later in Phase B before any ONNX productness export ships.

Default I'll use unless you say otherwise: **(a)**, 10% holdout, deterministic seed, paths excluded from train.

→ Approve "(a) default" or pick "(b)" and I'll move to Phase 2 (plan) and Phase 3 (tasks).
