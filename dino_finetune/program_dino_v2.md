# DINOv3 Fine-tuning V2 -- Agent Instructions

This is the autonomous experimentation guide for **V2** DINOv3 ViT-H+ LoRA fine-tuning. V2 extends V1 (`program_dino.md`) with:

- **New data sources**: `REID_COMMODITY_DIR` (30k flat unlabeled images, SSL-only) and `REID_NEGATIVES_DIR` (14 heterogeneous pseudo-classes, masked InfoNCE denominators only).
- **New losses**: ArcFace with phase-out, SSL consistency (true two-view independent augmentations), masked InfoNCE for negatives, optional base-DINOv3 anchor loss.
- **Stronger augmentation**: ColorJitter / GaussianBlur / RandomErasing.
- **Full feature-flag ablation surface**: every new technique can be toggled independently.

You are fine-tuning DINOv3 ViT-H+ (840M parameters, 1280-d embeddings) via LoRA. Your goal: maximize `combined_metric = 0.5 * recall@1 + 0.5 * mean_cosine` on the held-out val set. V1 baseline to beat: **combined=0.8094** (commit `43c0239`).

## Agent Configuration

This experiment loop should be run by a specialized **AI Engineer agent** (`voltagent-data-ai:ai-engineer`). The agent brings deep knowledge of:

- Contrastive & metric learning (InfoNCE, ArcFace, SupCon)
- LoRA / PEFT fine-tuning of large vision transformers
- Self-supervised consistency regularization (SimCLR / MoCo / SupCon patterns)
- Multi-source data curation (labeled + unlabeled + hard negatives)
- Hyperparameter search strategies and ablation design

Treat this as a research project, not a checklist. Form hypotheses, test them, analyze results, and adapt strategy based on what the data reveals.

## Setup

1. **Agree on a run tag** with the user (e.g. `dino-v2-apr25`). Branch `autoresearch/<tag>` must not exist yet.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current head.
3. **Read in-scope files**:
   - `dino_finetune/program_dino_v2.md` -- these instructions (you are reading it)
   - `dino_finetune/program_dino.md` -- V1 instructions for cross-reference and inherited knowledge
   - `dino_finetune/prepare_dino.py` -- **IMMUTABLE**: model loading, data pipeline, evaluation, adapter save/load. Do not modify.
   - `dino_finetune/train_dino.py` -- **V1 CODE, IMMUTABLE in v2 loop**: preserves V1 optimal config. Do not modify.
   - `dino_finetune/train_dino_v2.py` -- **YOUR FILE**: feature flags, losses, dataset, augmentation, optimizer. Fair game within constraints.
4. **Initialize results_v2.tsv** if missing. Header (7 TAB-separated columns):
   ```
   commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
   ```
5. **Run baseline**: `cd dino_finetune && python train_dino_v2.py > run_v2.log 2>&1`. First run is always the unmodified `train_dino_v2.py` (all flags ON) -- this establishes the V2 baseline / ceiling.
6. **Record baseline** in `results_v2.tsv`, status=`baseline`, description=`V2 baseline: all flags on`. Commit as baseline.
7. **Begin experiment loop**.

## Hard Constraints -- NEVER VIOLATE

1. **NEVER edit `prepare_dino.py`** -- it contains data loading, model loading, evaluation, and adapter I/O. Modifying breaks the trust boundary and makes experiments non-comparable.

2. **NEVER edit `train_dino.py`** -- it preserves V1 experiment history & the configuration that produced combined=0.8094. V2 writes to `train_dino_v2.py` only.

3. **NEVER edit `results.tsv`** -- it is V1's log. V2 writes to `results_v2.tsv` only.

4. **NEVER write V2 checkpoints into V1 directories.** V2 output paths:
   - `output/best_adapter_v2/`
   - `output/last_adapter_v2/`
   - `output/last_adapter_v2/checkpoint.pt`

   V1 paths (`output/best_adapter/`, `output/last_adapter/`) must remain untouched.

5. **NEVER add new dependencies** -- only: `peft`, `transformers`, `torch`, `torchvision`, `numpy`, `PIL`, `loguru`. Implement missing functionality inline in `train_dino_v2.py`.

6. **NEVER exceed `EPOCHS=30`** per experiment. V1 used 10-20. V2 default is 20. You may tune within [10, 30].

7. **NEVER stop the loop** -- run until manually interrupted. Do NOT ask "should I continue?" or "is this a good place to pause?". The human may be asleep. See NEVER STOP section below.

8. **NEVER disable gradient checkpointing** (`USE_GRADIENT_CHECKPOINTING = True`). DINOv3 ViT-H+ is 840M params; without checkpointing you WILL hit OOM on 24GB VRAM, especially with two-view SSL.

9. **NEVER set `BATCH_SIZE > 16`** -- VRAM ceiling on RTX 4090 (24GB). Use `GRADIENT_ACCUMULATION_STEPS` to get larger effective batches. With SSL on, effectively doubles forward VRAM.

10. **NEVER `USE_BASE_ANCHOR=True` together with `BATCH_SIZE > 8`** -- base anchor loads a SECOND frozen 840M model in memory. Only safe at small batches.

## Search Space -- Experiment Variables

All tunables live at the top of `train_dino_v2.py`. Read the file to confirm exact names before editing.

### V2 Feature Flags (NEW -- the primary ablation surface)

| Flag | Default | What it enables |
|------|---------|-----------------|
| `USE_ARCFACE` | `True` | ArcFace angular-margin head (supervised classification loss on product classes). Phased out across epochs. |
| `USE_SSL_CONSISTENCY` | `True` | Two independent augmented views per image; pulls their embeddings together. Dataset returns `(view_a, view_b, label)`. Doubles forward VRAM. |
| `USE_NEGATIVES_MASKED_NCE` | `True` | Include negatives/commodity in the InfoNCE denominator (as hard distractors). When `False`, they are filtered out entirely before the loss. |
| `USE_STRONG_AUG` | `True` | ColorJitter + GaussianBlur + RandomErasing + wider `RandomResizedCrop` scale. |
| `USE_BASE_ANCHOR` | `False` | Anchor LoRA embeddings to frozen base DINOv3 (prevents catastrophic drift). Doubles forward compute; requires small batch. |

### Loss Weights

| Constant | Default | Safe Range | Notes |
|----------|---------|------------|-------|
| `ARCFACE_WEIGHT` | 0.1 | [0.0, 1.0] | Coefficient on ArcFace (decays linearly to 0 by `ARCFACE_PHASEOUT_EPOCH`). |
| `ARCFACE_SCALE` | 30.0 | [16, 64] | Cosine scaling inside ArcFace. |
| `ARCFACE_MARGIN` | 0.3 | [0.1, 0.5] | Angular margin in radians. |
| `ARCFACE_PHASEOUT_EPOCH` | 5 | [0, EPOCHS] | Epoch by which ArcFace weight reaches 0. 0 = no phase-out. |
| `SSL_WEIGHT` | 0.3 | [0.0, 1.0] | Coefficient on `1 - cos(view_a, view_b)` alignment loss. |
| `BASE_ANCHOR_WEIGHT` | 0.1 | [0.0, 0.5] | Coefficient on `1 - cos(lora_emb, base_emb)` when `USE_BASE_ANCHOR=True`. |

### InfoNCE Core

| Constant | Default | Safe Range | What it controls |
|----------|---------|------------|------------------|
| `TEMPERATURE` | 0.20 | [0.03, 0.30] | **CRITICAL** -- InfoNCE temperature. V1 found 0.20 (soft) beat 0.07 on combined. Lower = sharper (recall ↑, cosine ↓). Higher = softer (cosine ↑, recall ↓ or collapse). |
| `BATCH_SIZE` | 8 | [4, 16] | Physical batch size. Two-view SSL doubles VRAM. |
| `GRADIENT_ACCUMULATION_STEPS` | 16 | [4, 64] | Effective batch = BATCH_SIZE × this. |
| `LR` | 5e-4 | [1e-5, 1e-3] | AdamW learning rate. V1 found 5e-4 on the edge of divergence; try 2e-4 or 3e-4 if collapse appears. |
| `WEIGHT_DECAY` | 0.01 | [0.0, 0.1] | AdamW weight decay. |
| `WARMUP_RATIO` | 0.2 | [0.05, 0.4] | Fraction of steps for linear warmup. V2 defaults higher (0.2) because more losses interact. |

### LoRA Config

| Constant | Default | Safe Range | Notes |
|----------|---------|------------|-------|
| `LORA_R` | 16 | [4, 64] | Rank. Higher = more capacity + VRAM. V1 found 16 optimal. |
| `LORA_ALPHA` | 32 | [8, 128] | Scaling (typically 2×R). Effective scale = ALPHA / R. |
| `LORA_DROPOUT` | 0.05 | [0.0, 0.2] | Regularization. |
| `LORA_TARGET_MODULES` | `["q_proj", "v_proj"]` | see below | Which attention projections get LoRA adapters. |

**`LORA_TARGET_MODULES` expansion guide** (same as V1):
- Default `["q_proj", "v_proj"]` -- safe, ~0.2% trainable params
- `+ "k_proj"` -- marginal capacity, low risk
- `+ "out_proj"` -- moderate capacity increase
- `+ "mlp.fc1", "mlp.fc2"` -- significant increase, watch VRAM
- Full expansion with high LORA_R will OOM

### Data Caps & Ratios (for `V2MultiSourceDataset`)

| Constant | Default | Safe Range | Notes |
|----------|---------|------------|-------|
| `RETAIL_MAX_PER_CLASS` | 100 | [50, 300] | Cap retail samples per class. V1's commodity-equivalent. |
| `COMMODITY_MAX_SAMPLES` | 20000 | [5000, 30000] | Cap unlabeled commodity pool (full 30k overwhelms products). |

### Augmentation Pipeline (`build_v2_train_transform`)

When `USE_STRONG_AUG=True`:
```
RandomResizedCrop(size, scale=(0.4, 1.0))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
RandomApply([GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5)
ToTensor + Normalize
RandomErasing(p=0.25)
```

When `USE_STRONG_AUG=False`: V1-style (RandomResizedCrop scale=(0.5, 1.0) + HFlip + ToTensor + Normalize).

### Early Stopping / Safety

| Constant | Default | Notes |
|----------|---------|-------|
| `EARLY_STOP_COSINE_THRESHOLD` | 0.95 | Collapse detector (cosine too high → over-clustered / degenerate). |
| `EARLY_STOP_COLLAPSE_CONSECUTIVE` | 3 | Stop after this many consecutive collapsed epochs. |
| `EARLY_STOP_PATIENCE` | 10 | Stop if `combined` hasn't improved this many epochs. |
| `EARLY_STOP_RECALL_DROP` | 0.15 | Stop if recall@1 drops this much from best. |
| `USE_GRADIENT_CHECKPOINTING` | `True` | **NEVER False** (OOM). |

## Experiment Strategy (Prioritized)

Baseline first (all flags ON), then ablate. The V2 baseline run tells you the ceiling; ablations tell you what's actually contributing.

### Priority 1: Feature-Flag Ablation (find what actually helped)

After baseline, turn off ONE flag at a time. Any flag whose removal significantly hurts = keep; any flag whose removal doesn't hurt = remove to save compute.

**Suggested experiments:**
- `USE_SSL_CONSISTENCY = False` (keeps single-view; saves ~50% forward VRAM)
- `USE_ARCFACE = False` (InfoNCE-only)
- `USE_NEGATIVES_MASKED_NCE = False` (fully remove negatives from batch)
- `USE_STRONG_AUG = False` (back to V1-style aug)

Log each as `ablation: <flag> off`. Compare to baseline delta.

### Priority 2: Temperature Tuning (MOST IMPACTFUL single knob)

V1 found T=0.20 (soft) beats T=0.07 (sharp). But V2 has more losses interacting — retest.

**Suggested experiments:**
- `TEMPERATURE = 0.15`
- `TEMPERATURE = 0.25`
- `TEMPERATURE = 0.10` (sharp; watch for cosine collapse)
- `TEMPERATURE = 0.30` (very soft; watch for recall drop)

If loss goes NaN or cosine explodes → temperature too low. Back off.

### Priority 3: Loss Weight Sweeps

After finding the best single-flag configuration, tune its weight.

**ArcFace (if `USE_ARCFACE=True`):**
- `ARCFACE_WEIGHT = 0.05, 0.2, 0.3, 0.5`
- `ARCFACE_PHASEOUT_EPOCH = 3, 8, 10`
- `ARCFACE_MARGIN = 0.2, 0.4, 0.5`
- `ARCFACE_SCALE = 20, 50, 64`

**SSL (if `USE_SSL_CONSISTENCY=True`):**
- `SSL_WEIGHT = 0.1, 0.5, 1.0`

**Base anchor (for stability-sensitive configs):**
- `USE_BASE_ANCHOR = True` with `BASE_ANCHOR_WEIGHT = 0.05, 0.1, 0.2`
- Warning: requires `BATCH_SIZE ≤ 8` (two models in VRAM).

### Priority 4: Augmentation Depth

Beyond `USE_STRONG_AUG` on/off, edit `build_v2_train_transform()` and experiment creatively:

1. **ColorJitter strength** — hue ∈ {0.0, 0.05, 0.15}
2. **RandomResizedCrop scale** — (0.2, 1.0) vs (0.6, 1.0) — tighter crops force fine-grained features
3. **RandomPerspective(distortion_scale=0.2, p=0.3)** — camera angle simulation
4. **RandomAffine(degrees=15, translate=(0.1, 0.1))** — rotation/translation
5. **RandomGrayscale(p=0.1)** — reduce color over-reliance
6. **Motion blur / defocus** — custom transform, simulate bad cameras
7. **RandomErasing scale/ratio** — (0.02, 0.33), various ratios

### Priority 5: LoRA Capacity

Test whether more/less LoRA adapter parameters help.

**Suggested:**
- `LORA_R=8, LORA_ALPHA=16` (reduced capacity)
- `LORA_R=32, LORA_ALPHA=64` (increased)
- `LORA_R=64, LORA_ALPHA=128` (high; watch VRAM)
- Expand modules: `LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj"]`
- Full attention: `+ "out_proj"`
- Full attention + MLP: `+ "mlp.fc1", "mlp.fc2"` (significant VRAM hit)

### Priority 6: Effective Batch Size

InfoNCE benefits from more negatives per anchor. Increase effective batch via accumulation.

**Suggested:**
- Effective 64: `BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=8`
- Effective 256: `BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=32`
- Effective 384: `BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=48`

Compensate: larger effective batch = fewer optimizer steps per epoch = may need higher LR or more epochs.

### Priority 7: Data Balance & Mixing

The dataset is a union: `TRAIN_DIR + REID_PRODUCTS_DIR + retail + commodity + negatives`. Too much commodity can wash out product signal.

**Suggested:**
- `COMMODITY_MAX_SAMPLES = 5000, 10000, 30000`
- `RETAIL_MAX_PER_CLASS = 50, 200`
- Edit `V2MultiSourceDataset` to skip negatives entirely: comment out `negatives_root=` — then ablation-compare to baseline.

### Priority 8: Advanced / Contrarian

After exhausting 1-7, try radical ideas:

- Replace InfoNCE with Supervised Contrastive (SupCon): treat same-class images across both views as positives (`concat([emb_a, emb_b])` with doubled labels).
- Use teacher-student self-distillation with LoRA as student and base as teacher (`USE_BASE_ANCHOR` with higher weight).
- Curriculum: start with `USE_STRONG_AUG=False` for 3 epochs, then `True` (you'd need to edit the training loop).
- Test-time augmentation: at eval, forward image at multiple augmented views and average embeddings.
- Freezing lower transformer blocks: extend LoRA to only deeper blocks (edit `LORA_TARGET_MODULES` to reference layer indices — requires custom regex).

## Workflow -- The Experiment Loop

LOOP FOREVER:

1. **Read history**: `cat dino_finetune/results_v2.tsv` AND `cat dino_finetune/results.tsv` for V1 reference. What has been tried? What improved? What patterns emerge? Compare V2 runs to V1 baseline.

2. **Choose next experiment**: Based on history + priority list. One idea per experiment for clear attribution. Prefer unexplored flag combinations over minor hyperparameter sweeps.

3. **Edit `train_dino_v2.py`**: Minimal, focused diff. One idea per commit.

4. **git commit**: Descriptive message (e.g., `v2: ablate SSL (USE_SSL_CONSISTENCY=False)`).

5. **Run**:
   ```bash
   cd dino_finetune && python train_dino_v2.py > run_v2.log 2>&1
   ```

6. **Read result**:
   ```bash
   grep "RESULT V2:\|METRIC:" dino_finetune/run_v2.log
   ```
   If empty → run crashed; `tail -n 50 dino_finetune/run_v2.log` for stack trace.

7. **Log to `results_v2.tsv`** (7 TAB-separated columns). Do NOT git-track `results_v2.tsv`.

8. **Keep or discard**:
   - `combined_metric` improved (higher than V2 best)? **KEEP** → advance branch.
   - Same or worse? **DISCARD** → `git reset --hard HEAD~1`.
   - Crash? Log as `crash`, `git reset --hard HEAD~1`.
   - 3+ consecutive crashes on same idea → SKIP that direction.

9. **GOTO 1**.

## Output Format

`train_dino_v2.py` logs at the end:

```
RESULT V2: recall@1=0.4321 mean_cosine=0.8765 combined=0.6543 peak_vram_mb=18432
METRIC: 0.654300
```

Extract:
```bash
grep "RESULT V2:\|METRIC:" dino_finetune/run_v2.log
```

Per-epoch training logs show the loss decomposition:

```
Epoch 3/20: loss=2.1234 nce=2.0123 arc=0.3456(w=0.060) ssl=0.1234 anchor=0.0000 time=245.3s
```

This tells you which loss components are active (nonzero) and how ArcFace phase-out is progressing (`w=` decays).

## Logging Results

`dino_finetune/results_v2.tsv` -- tab-separated, 7 columns:

```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
a1b2c3d	0.820000	0.850000	0.790000	7500.1	baseline	V2 baseline: all flags on
b2c3d4e	0.815000	0.840000	0.790000	5200.3	keep	ablate: USE_SSL_CONSISTENCY=False (SSL hurts slightly)
c3d4e5f	0.000000	0.000000	0.000000	0.0	crash	LORA_R=64 + all MLP modules (OOM)
```

Status values: `baseline`, `keep`, `discard`, `crash`.
Description: one-liner that tells future-you what you tried and why it was interesting.

## Crash Handling

1. **OOM** (`CUDA out of memory`):
   - Log `crash` in `results_v2.tsv`. Reset: `git reset --hard HEAD~1`.
   - Next experiment MUST reduce compute: lower `LORA_R`, disable `USE_SSL_CONSISTENCY` (halves forward), disable `USE_BASE_ANCHOR`, lower `BATCH_SIZE`.
   - If `peak_vram_mb > 22000` on last successful run, do NOT add any complexity.

2. **Bug** (typo, import, shape mismatch):
   - Not a failed experiment -- it's a code bug. `git reset --hard HEAD~1`, fix, re-commit, re-run.

3. **NaN loss**:
   - Almost always temperature too low. Increase `TEMPERATURE`.
   - Less commonly: `SSL_WEIGHT` too high with aggressive aug. Lower weight.
   - `git reset --hard HEAD~1`.

4. **Cosine collapse** (early-stop fires with `mean_cosine > 0.95`):
   - `ARCFACE_WEIGHT` too high with low temperature. Drop one or both.
   - `LR` too aggressive. Try half.
   - `USE_BASE_ANCHOR=True` can stabilize (at VRAM cost).

5. **3+ consecutive crashes on same direction**:
   - SKIP entirely. Move to a completely different priority.

## Domain Context: DINOv3 Contrastive Fine-tuning V2

- **DINOv3 ViT-H+** — 840M param vision transformer pretrained on LVD-1689M. 1280-d CLS token embeddings. In V2 we still fine-tune only via LoRA adapters (~0.2% trainable).

- **V2's loss landscape** — simultaneously optimizes:
  - **Masked InfoNCE** (primary contrastive signal on labeled products, negatives/commodity as denominators).
  - **ArcFace** (angular-margin classification head; phased out mid-training to avoid overfitting to closed-set).
  - **SSL consistency** (two-view embedding alignment; teaches augmentation invariance).
  - **Base anchor** (optional; regularizes toward frozen base DINOv3 to prevent drift).

- **Data composition** (per `V2MultiSourceDataset`):
  - `TRAIN_DIR` (product_code_dataset/train) — labeled
  - `REID_PRODUCTS_DIR` (reid_multiple/products) — labeled, large
  - `RETAIL_DIR` (retail_product_checkout_crop) — labeled, per-class capped
  - `REID_COMMODITY_DIR` (flat, unlabeled) — label=-1, SSL only
  - `REID_NEGATIVES_DIR` (heterogeneous, pseudo-labeled) — label=-2, InfoNCE denominator only

- **Combined metric** — `0.5 * recall@1 + 0.5 * mean_cosine`. Same as V1 for apples-to-apples comparison.

- **Why V2?** — V1 topped out at combined=0.8094 with InfoNCE only. V2 adds supervised class signal (ArcFace), invariance regularization (SSL), hard-negative structure (masked NCE with negatives), and drift protection (base anchor). Ablation tells us which combination moves the needle for product retrieval on this dataset.

- **Why fine-tune DINOv3?** — The v2 adapter becomes a stronger teacher for the lightweight LCNet student in `student_finetune/`. Better embeddings → better distillation → better deployable student.

## Reading results_v2.tsv for History Reasoning

Before EVERY experiment, read the tsv and analyze:

```bash
cat dino_finetune/results_v2.tsv
```

- **How many V2 experiments ran?** Early = explore broadly (feature flags). Mid = refine loss weights. Late = radical ideas.
- **What's the current V2 best?** What config produced it? What's the gap to V1 baseline (0.8094)?
- **Which flag ablations hurt most?** That flag is carrying weight — keep it in future configs.
- **Which flag ablations didn't hurt?** That flag may be redundant — drop it to save compute.
- **Recall vs cosine**: if recall↑ but cosine↓, you're at `TEMPERATURE` too low or too-aggressive ArcFace. If cosine↑ but recall↓, you may be collapsing — back off on pull-together losses (SSL, ArcFace) or raise temperature.
- **VRAM trends**: if `peak_vram_mb > 22000`, be cautious about adding any compute.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human. The human may be asleep. You are autonomous.

If you run out of ideas from the priority list:

1. **Re-read `results_v2.tsv` + `results.tsv`** for patterns across V1 AND V2.
2. **Re-read `train_dino_v2.py`** line by line for overlooked constants or code paths.
3. **Re-read this program_dino_v2.md** from the top; re-read `program_dino.md` for V1 ideas you haven't tried.
4. **Combine top-3 individual improvements** — 2-way and 3-way combinations.
5. **Contrarian experiments** — if all improvements go one direction, try the opposite extreme.
6. **Ablation studies** — remove one component at a time from current best.
7. **Search for novel approaches** — use WebSearch to find recent papers on:
   - LoRA + contrastive learning for retrieval
   - Self-supervised fine-tuning for retail/product retrieval
   - Hard-negative mining & masked contrastive losses
   - Feature-based distillation between ViTs at different scales
   - Curriculum / progressive augmentation schedules

Remember: you are an autonomous AI researcher. Do not just follow a checklist — THINK about what the data is telling you and form hypotheses. The best experiments come from understanding WHY something worked, not just WHAT worked.

## Convergence Signal & Final Training Readiness

Every 10 V2 experiments, check:

```
📊 V2 convergence check (experiment {N}):
  - Best combined_metric: {X} (from experiment {best_exp})
  - V1 baseline: 0.8094 — delta: {X - 0.8094:+.4f}
  - Last 5 improvements: {list deltas}
  - Unexplored priority levels: {list}
```

**Signal convergence when:**
- `combined_metric` hasn't improved > 0.005 for 10 consecutive kept experiments, AND
- All 8 priority levels have been explored, AND
- Best flag combinations and their 2-way interactions have been tried.

When convergent, print:

```
🏁 V2 SEARCH CONVERGED after {N} experiments.
Best configuration: recall@1={X}, mean_cosine={Y}, combined={Z}
Best commit: {hash}
V1 baseline beaten by: {Z - 0.8094:+.4f}

V2 FINAL TRAINING READY:
1. Stop this autoresearch loop
2. Change EPOCHS from 20 to 100 (or more) in train_dino_v2.py
3. Run: cd dino_finetune && python train_dino_v2.py
4. Best adapter will be saved to output/best_adapter_v2/
5. Feed it to student_finetune's DINOv3FTTeacher by pointing adapter_path there.
```

**Then keep experimenting** (NEVER STOP). The convergence signal is informational for the human. You keep looking for late-stage breakthroughs.
