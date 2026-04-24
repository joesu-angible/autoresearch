# Student Distillation — V2 Protocol

This document extends `program_final_student.md` for the **V2 training run**.
V2 adds `TRAIN_DIR` (48k in-distribution images previously unused) and fixes `commodity` flat-file handling (30k images previously silently dropped).

**Read `program_final_student.md` and `program_student.md` first for the general protocol.**

## V1 baseline to beat

| Metric | V1 best | Commit |
|---|---|---|
| combined | **0.8588** | `87766d9` |
| recall@1 | 0.9006 | - |
| recall@5 | 0.9432 | - |
| mean_cosine | 0.8169 | - |

Goal: beat combined=0.8588.

## File overrides (CRITICAL — do NOT touch V1 files)

| V1 (immutable) | V2 (edit these) |
|---|---|
| `train_final.py` | **`train_v2.py`** |
| `train.py` | (reused as library for `run_train_epoch`, `LCNet`, `ArcMarginProduct`) |
| `results.tsv` | **`results_v2.tsv`** |
| `workspace/output/distill_final_lcnet050/` | **`workspace/output/distill_final_lcnet050_v2/`** |

**Do NOT edit `train.py`, `train_final.py`, or `results.tsv`** — those preserve V1 experiment history and ablation-proven optimal config.
**Do NOT edit `prepare.py`** — it is the immutable trust boundary.

## What V2 changes

| Aspect | V1 | V2 |
|---|---|---|
| Primary data | REID_PRODUCTS + REID_COMMODITY (broken, flat dir skipped) | **TRAIN_DIR** + REID_PRODUCTS + REID_COMMODITY (flat-file handled) |
| Commodity | Silently dropped (`d.is_dir()` filter) | Loaded as pseudo-unlabeled, distillation only (not ArcFace) |
| Augmentation | Default (HFlip/VFlip/PadToSquare/Resize) | **ColorJitter/GaussianBlur/RandomErasing** (togglable) |
| Teacher cache | `teacher_cache/dinov3_ft/` | Same cache, but will be extended with new image paths (~50k+ new) |
| SSL consistency | N/A | Intentionally NOT added — distillation already teaches aug invariance implicitly |

## V2 feature flags

Edit at top of `train_v2.py`:

```python
USE_STRONG_AUG = True               # ColorJitter + GaussianBlur + RandomErasing
```

(ArcFace + phase-out + negatives blacklist all inherited from V1 optimal config.)

## Run

```bash
cd student_finetune
python train_v2.py --max-epochs 30 > run_v2.log 2>&1
```

First run will rebuild teacher cache for newly-added TRAIN_DIR + commodity images (~50k+ new paths). Allow ~30 min for cache build on first run. Subsequent runs reuse the cache.

Output printed at end:
- `V2 TRAINING COMPLETE`
- `Best: <float>` — the combined_metric to log

## Logging to results_v2.tsv

Append one row per experiment (9 columns matching V1 schema):

```
commit    combined_metric    recall_1    recall_5    mean_cosine    distill_loss    peak_vram_mb    status    description
```

First row: run with default V2 config → label as `baseline`, description="V2 baseline: added TRAIN_DIR + commodity flat-file handling + strong aug".

## Experiment loop

Same as `program_student.md`, but:
- Edit only `train_v2.py`.
- Log only to `results_v2.tsv`.
- Never touch V1 files or V1 tsv.

## Ablation priority (if V2 beats V1)

1. `USE_STRONG_AUG = False` → measures augmentation contribution.
2. Remove `TRAIN_DIR` from `primary_roots` in `build_v2_distill_dataset` → measures TRAIN_DIR contribution.
3. Set `commodity_ratio=0.0` in `build_v2_distill_dataset` → measures commodity contribution.
4. Sweep `commodity_ratio` and `retail_ratio` values.

## Constraints

- Never add new dependencies.
- Never edit `prepare.py`, `train.py`, `train_final.py`, `results.tsv`.
- Commit after each experiment.
- Use `/home/whiskey/workspace/project/central/v2/training/autoresearch/.venv/bin/python` as python.
