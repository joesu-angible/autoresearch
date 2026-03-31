# Student Distillation -- Agent Instructions

This is the autonomous experimentation guide for LCNet student model training via knowledge distillation from multiple teacher models. You are training a lightweight LCNet backbone (256-dimensional embeddings) distilled from large teacher models (TrendyolONNX, DINOv2, DINOv3-FT, C-RADIO). Your goal: maximize recall@1 on the validation set.

## Agent Configuration

This experiment loop should be run by a specialized **AI Engineer agent** (`voltagent-data-ai:ai-engineer`). The agent brings deep knowledge of:
- Knowledge distillation techniques and loss function design
- Model architecture optimization under deployment constraints
- Hyperparameter search strategies and experiment design
- Reading and interpreting training metrics to form hypotheses

The agent should treat this as a research project, not a checklist. Form hypotheses, test them, analyze results, and adapt strategy based on what the data reveals.

## Setup

To set up a new student distillation experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `student-mar31`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The sub-project is self-contained. Read these files for full context:
   - `student_finetune/program_student.md` -- these instructions (you are reading it)
   - `student_finetune/prepare.py` -- IMMUTABLE: data pipeline, teacher definitions, evaluation, cache management. Do not modify.
   - `student_finetune/train.py` -- YOUR FILE: model architecture, loss functions, optimizer, teacher selection, augmentation. Everything here is fair game within constraints.
4. **Pre-build teacher caches** (optional but recommended):
   ```bash
   cd student_finetune && python build_caches.py
   ```
   This pre-computes teacher embeddings so training doesn't stall on inference. Skip if caches already exist.
5. **Initialize results.tsv** in the `student_finetune/` directory with just the header row:
   ```
   commit	combined_metric	recall_1	recall_5	mean_cosine	distill_loss	peak_vram_mb	status	description
   ```
6. **Run baseline**: `cd student_finetune && python train.py > run.log 2>&1` -- your first run is always the unmodified train.py to establish the baseline.
7. **Record baseline** in results.tsv, commit as baseline.
8. **Begin experiment loop**.

## Hard Constraints -- NEVER VIOLATE

These are absolute rules. Violating any one invalidates all experiments.

1. **NEVER edit prepare.py** -- it contains data loading, teacher inference/caching, evaluation, and metric computation. Modifying it breaks the trust boundary and makes all experiments non-comparable.

2. **NEVER add new dependencies** -- only use what is already available: `torch`, `torchvision`, `timm`, `numpy`, `PIL`, `loguru`, `onnxruntime`. Implement missing features yourself in train.py.

3. **NEVER exceed EPOCHS=10 per experiment** -- this is the fixed experiment budget enforced by prepare.py. You optimize WHAT happens in 10 epochs, not how many epochs to train.

4. **NEVER stop the loop** -- run until manually interrupted. The human may be asleep.

5. **NEVER set BATCH_SIZE > 512** -- physical batch size is limited by VRAM on RTX 4090 (24GB). Distillation batch of 256 is default and safe.

6. **NEVER increase LCNET_SCALE above 0.5** -- the student model must deploy on edge devices. The baseline `timm/lcnet_050.ra2_in1k` (~1.88M params, ~39M FLOPs at 224x224) is the hard ceiling. Any architecture change that pushes total params or FLOPs above the lcnet_050 baseline is forbidden. This means LCNET_SCALE=1.0 and LCNET_SCALE=1.5 are permanently off-limits.

## Search Space -- Experiment Variables

These are all the tunable constants at the top of `train.py`. Read the file to confirm exact variable names before editing.

### Teacher Selection (MOST IMPACTFUL)

| Constant | Default | Options | What It Controls |
|----------|---------|---------|------------------|
| `TEACHER` | `"trendyol_onnx"` | See registry | Single teacher mode (backward compatible) |
| `TEACHERS` | `None` | `dict[str, float]` | Multi-teacher mode with per-teacher weights. Set to `None` to use `TEACHER` instead. |

**Available teachers** (from TEACHER_REGISTRY):
- `"trendyol_onnx"` -- 256d, ONNX quantized, fast inference. Pre-existing cache.
- `"dinov2"` -- 256d, Trendyol DINOv2 ecommerce model
- `"dinov3_ft"` -- 1280d, Fine-tuned DINOv3 ViT-H+ with LoRA adapter (our best teacher)
- `"radio_so400m"` -- Dynamic dim, C-RADIOv4 SO400M with adaptors
- `"radio_h"` -- Dynamic dim, C-RADIOv4 Huge with adaptors

**Multi-teacher example:**
```python
TEACHERS = {
    "trendyol_onnx": 0.3,
    "dinov3_ft": 0.7,
}
```

### Training Hyperparameters

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `LR` | 2e-3 | [1e-4, 1e-2] | Learning rate for AdamW. Most impactful after teacher selection. |
| `BATCH_SIZE` | 256 | [64, 512] | Distillation batch size. Larger = better but more VRAM. |
| `WEIGHT_DECAY` | 1e-5 | [0, 1e-3] | AdamW weight decay regularization. |
| `BACKBONE_LR_MULT` | 0.1 | [0.01, 1.0] | Backbone LR multiplier. Lower = more stable, higher = faster adaptation. |
| `DROP_HARD_RATIO` | 0.2 | [0.0, 0.5] | Fraction of hardest negative samples dropped from loss. Prevents noisy gradients. |
| `QUALITY_DEGRADATION_PROB` | 0.5 | [0.0, 1.0] | Probability of applying quality degradation (blur, JPEG compression) to training images. |

### ArcFace Metric Learning

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `USE_ARCFACE` | `True` | bool | Enable/disable ArcFace auxiliary loss |
| `ARCFACE_S` | 32.0 | [16, 64] | ArcFace scale parameter |
| `ARCFACE_M` | 0.50 | [0.1, 0.7] | ArcFace angular margin |
| `ARCFACE_LOSS_WEIGHT` | 0.03 | [0.0, 0.2] | Weight of ArcFace loss relative to distillation loss |
| `ARCFACE_BATCH_SIZE` | 128 | [64, 256] | Separate batch size for ArcFace data |
| `ARCFACE_MAX_PER_CLASS` | 100 | [50, 200] | Max samples per class in ArcFace dataset |
| `ARCFACE_PHASEOUT_EPOCH` | 0 | [0, 10] | Epoch to disable ArcFace (0 = never disable) |

### LCNet Architecture

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `LCNET_SCALE` | 0.5 | [0.35, 0.5] | Width multiplier. **HARD LIMIT: never exceed 0.5** (edge deployment constraint). 0.35 = smaller/faster. |
| `SE_START_BLOCK` | 10 | [0, 12] | Block index where Squeeze-and-Excite begins (0-indexed, 13 total blocks). Lower = more SE. |
| `SE_REDUCTION` | 0.25 | [0.125, 0.5] | SE squeeze ratio. Lower = more capacity. |
| `ACTIVATION` | `"h_swish"` | `"h_swish"`, `"relu"`, `"gelu"` | Activation function for LCNet blocks. |
| `KERNEL_SIZES` | `[3,...,5]` | per-block 3 or 5 | Per-block kernel sizes (13 blocks). Larger kernels = larger receptive field. |
| `USE_PRETRAINED` | `True` | bool | Load timm pretrained weights (only when LCNET_SCALE matches a known model). |

### Self-Supervised Learning (SSL)

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `SSL_WEIGHT` | 0.0 | [0.0, 0.2] | SSL contrastive loss weight. **WARNING: enabling doubles forward passes (VRAM ~1.5x).** |
| `SSL_TEMPERATURE` | 0.07 | [0.03, 0.2] | InfoNCE temperature for SSL loss |
| `SSL_PROJ_DIM` | 128 | [64, 256] | SSL projection head output dimension |

### RADIO Teacher Settings

| Constant | Default | Options | What It Controls |
|----------|---------|---------|------------------|
| `RADIO_VARIANT` | `"so400m"` | `"so400m"`, `"h"` | C-RADIOv4 model size |
| `RADIO_ADAPTORS` | `["backbone"]` | subset of `["backbone", "dino_v3_7b", "siglip2-g", "sam3"]` | Which RADIO adaptors to distill from |
| `SPATIAL_DISTILL_WEIGHT` | 0.0 | [0.0, 1.0] | Spatial (per-patch) distillation from RADIO. 0 = disabled. |

### Advanced Training Techniques

| Constant | Default | What It Controls |
|----------|---------|------------------|
| `ENABLE_PHI_S` | `False` | PHI-S Hadamard transform -- prevents dominant teacher from overwhelming gradients |
| `ENABLE_FEATURE_NORMALIZER` | `False` | Per-teacher online whitening during warmup |
| `ENABLE_ADAPTOR_MLP_V2` | `False` | Enhanced projection heads (LayerNorm + GELU + 2-layer MLP) |
| `ENABLE_L_ANGLE` | `False` | Angular dispersion normalization for fair multi-teacher weighting |
| `ENABLE_HYBRID_LOSS` | `False` | Cosine + Smooth-L1 hybrid loss for spatial features |
| `ENABLE_FEATSHARP` | `False` | Feature sharpening |
| `ENABLE_SHIFT_EQUIVARIANT` | `False` | Shift equivariant loss |
| `VAT_WEIGHT` | 0.0 | Virtual Adversarial Training regularization |
| `SEP_WEIGHT` | 1.0 | Separation loss weight |

## Experiment Strategy (Prioritized)

Work through these priorities in order. Exhaust each priority level before moving to the next.

### Priority 1: Teacher Selection (MOST IMPACTFUL)

The teacher signal is the foundation of distillation. A better teacher = better student regardless of other settings.

**Suggested experiments:**
- `TEACHER = "trendyol_onnx"` (baseline, 256d)
- `TEACHER = "dinov3_ft"` (1280d, our fine-tuned DINOv3 -- likely best single teacher)
- Multi-teacher: `TEACHERS = {"trendyol_onnx": 0.5, "dinov3_ft": 0.5}`
- Multi-teacher: `TEACHERS = {"trendyol_onnx": 0.3, "dinov3_ft": 0.7}` (weight toward stronger teacher)
- All teachers: `TEACHERS = {"trendyol_onnx": 0.2, "dinov2": 0.2, "dinov3_ft": 0.6}`

**Note:** Each teacher needs cached embeddings. Run `python build_caches.py <teacher_name>` first.

### Priority 2: Learning Rate + Batch Size

After finding the best teacher, tune LR and batch size.

**Suggested experiments:**
- LR=1e-3 (conservative)
- LR=5e-3 (aggressive)
- BATCH_SIZE=128 (smaller, more optimizer steps per epoch)
- BATCH_SIZE=512 (larger, more stable gradients)
- BACKBONE_LR_MULT=0.01 (freeze backbone nearly)
- BACKBONE_LR_MULT=0.5 (train backbone more)

### Priority 3: ArcFace Tuning

ArcFace adds explicit class boundaries. Test the balance.

**Suggested experiments:**
- `USE_ARCFACE = False` (distillation only -- compare)
- `ARCFACE_LOSS_WEIGHT = 0.1` (stronger ArcFace)
- `ARCFACE_M = 0.3` (easier margin)
- `ARCFACE_M = 0.7` (harder margin)
- `ARCFACE_PHASEOUT_EPOCH = 5` (ArcFace first half, then distillation only)

### Priority 4: LCNet Architecture (Iso-Parameter Experiments)

Explore architecture changes that improve quality WITHOUT increasing params/FLOPs beyond the lcnet_050 baseline (~1.88M params, ~39M FLOPs at 224x224). **LCNET_SCALE must NEVER exceed 0.5.**

**Suggested experiments (iso-parameter or parameter-reducing):**

1. **Smaller model (reduces params):**
   - `LCNET_SCALE = 0.35` -- ~0.9M params (~52% fewer). Tests whether a smaller model with better training signal can match lcnet_050.

2. **SE block placement (minimal param change):**
   - `SE_START_BLOCK = 6` -- adds SE to blocks 6-12 instead of 10-12. At scale=0.5, this adds ~25K params (SE on 128-ch blocks: 2 x Conv1x1 with reduction=0.25 = ~8.4K per block). Total stays well under 1.88M.
   - `SE_START_BLOCK = 0` -- SE on all 13 blocks. Adds ~35K total params (early blocks have 8-32 channels, so SE overhead is tiny: <1K per block). Still under baseline.
   - `SE_START_BLOCK = 8` -- SE on blocks 8-12 only. Moderate SE coverage, ~17K extra params.

3. **SE squeeze ratio (minimal param change):**
   - `SE_REDUCTION = 0.125` -- halves SE mid-channels, removes ~25K params from SE blocks. Tests if SE with less capacity still helps.
   - `SE_REDUCTION = 0.5` -- doubles SE mid-channels, adds ~25K params. More SE capacity, still well under baseline ceiling.

4. **Activation function (zero param change, different compute profile):**
   - `ACTIVATION = "gelu"` -- smoother gradient flow than h_swish. Same params, slightly higher FLOPs (~5% due to exp/erf), still well under baseline.
   - `ACTIVATION = "relu"` -- fastest activation, fewer FLOPs than h_swish. Tests if simpler activation is sufficient with good distillation.

5. **Kernel sizes (zero param change, modest FLOP change):**
   - `KERNEL_SIZES = [5,5,5,5,5,5,5,5,5,5,5,5,5]` -- all 5x5 depthwise kernels. Adds ~0 params (depthwise conv weights scale with kernel area but are tiny: +16 weights per channel per block). FLOPs increase ~2.7x on depthwise layers, but depthwise is <5% of total FLOPs, so net impact is <15% total FLOP increase. Still under baseline.
   - `KERNEL_SIZES = [3,3,3,3,3,3,3,3,3,5,5,5,5]` -- larger kernels only in later stages where receptive field matters most. Negligible FLOP/param change.
   - `KERNEL_SIZES = [5,5,5,3,3,3,3,3,3,5,5,5,5]` -- larger kernels in early + late stages (capture both low-level texture and high-level structure).

6. **Combined iso-param experiments (after individual winners identified):**
   - Best SE_START_BLOCK + best ACTIVATION + best KERNEL_SIZES -- combine winners from above.

### Priority 5: Advanced Techniques

Only try these after Priorities 1-4 are exhausted.

**Suggested experiments:**
- `SSL_WEIGHT = 0.05` (self-supervised auxiliary loss)
- `ENABLE_PHI_S = True` (only useful with multi-teacher)
- `ENABLE_FEATURE_NORMALIZER = True` (only useful with multi-teacher)
- `SPATIAL_DISTILL_WEIGHT = 0.1` (requires RADIO teacher)
- `ENABLE_ADAPTOR_MLP_V2 = True` (enhanced projection heads)
- `VAT_WEIGHT = 0.01` (adversarial regularization)

### Priority 6: Data Augmentation

**Suggested experiments:**
- `QUALITY_DEGRADATION_PROB = 0.0` (no degradation)
- `QUALITY_DEGRADATION_PROB = 0.8` (heavy degradation)
- `DROP_HARD_RATIO = 0.0` (keep all negatives)
- `DROP_HARD_RATIO = 0.4` (drop more hard negatives)

## Workflow -- The Experiment Loop

LOOP FOREVER:

1. **Read history**: `cat student_finetune/results.tsv`. What has been tried? What improved? What patterns emerge?

2. **Choose next experiment**: Based on history, pick the next experiment from the priority list above. One idea per experiment for clear attribution.

3. **Edit train.py**: Make your changes. Keep diffs minimal and focused.

4. **git commit**: Commit the change with a descriptive message.

5. **Run**: `cd student_finetune && python train.py > run.log 2>&1`

6. **Read results**:
   ```bash
   grep "combined_metric:\|recall@1:\|status:" student_finetune/run.log
   ```
   If empty, the run crashed -- `tail -n 50 student_finetune/run.log` for the stack trace.

7. **Log to results.tsv**: Record all columns. Do NOT git-track results.tsv.

8. **Keep or discard**:
   - combined_metric improved? **KEEP** -- advance the branch.
   - Same or worse? **DISCARD** -- `git reset --hard HEAD~1`
   - Crash? Log as crash, `git reset --hard HEAD~1`

9. **GOTO 1**

## Output Format

After each run, the training script prints:

```
---
status:           success
combined_metric:  0.654321
recall@1:         0.432100
recall@5:         0.567800
mean_cosine:      0.876543
distill_loss:     0.012345
arc_loss:         0.003456
vat_loss:         0.000000
sep_loss:         0.001234
---
```

It also writes `metrics.json` with full results.

## Logging Results

Log every experiment to `student_finetune/results.tsv` (tab-separated):

```
commit	combined_metric	recall_1	recall_5	mean_cosine	distill_loss	peak_vram_mb	status	description	next_step
a1b2c3d	0.654321	0.432100	0.567800	0.876543	0.012345	18432.1	keep	baseline	try dinov3_ft teacher (stronger signal)
b2c3d4e	0.712345	0.860200	0.900000	0.564500	0.298000	980.2	keep	dinov3_ft teacher	try multi-teacher 0.5/0.5 blend
```

The `next_step` column records your reasoning for the NEXT experiment — what you plan to try and why. This creates a decision trail that helps you (and future runs) understand the experimental logic.

## Crash Handling

1. **OOM**: Reduce BATCH_SIZE or disable SSL (SSL doubles VRAM).
2. **NaN loss**: Usually temperature too low in SSL, or LR too high.
3. **Cache miss**: Run `python build_caches.py <teacher>` to pre-build caches.
4. **3+ consecutive crashes**: Skip that direction entirely.

## Domain Context: Student Distillation

- **LCNet**: Lightweight classification network from PaddlePaddle. 256-dimensional embeddings. Very fast inference (~1ms).
- **Knowledge Distillation**: Student learns to produce embeddings similar to teacher's. Teacher embeddings are pre-cached for efficiency.
- **Multi-Teacher Distillation**: Student learns from multiple teachers simultaneously. Per-teacher projection heads adapt student features to each teacher's embedding space.
- **TEACHER_REGISTRY**: Central registry in prepare.py mapping teacher names to classes, dimensions, and cache paths.
- **Combined metric**: `0.5 * recall@1 + 0.5 * mean_cosine`. Same as DINOv3 fine-tuning.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous.

If you run out of ideas from the priority list:

1. **Analyze results.tsv deeply**: Look for patterns across ALL experiments. Which metric is the bottleneck (recall vs cosine)? Which changes had the biggest delta? Are there unexplored interactions between variables?
2. **Hypothesize from data**: If recall is high but cosine is low, the model retrieves well but embeddings aren't tight — try stronger distillation weight or different loss. If cosine is high but recall is low, embeddings are similar but not discriminative — try ArcFace or harder negatives.
3. **Search for novel approaches**: Use WebSearch to find recent papers on knowledge distillation, lightweight model training, or metric learning improvements. Look for techniques like:
   - Progressive distillation (start with easy teacher, switch to hard)
   - Curriculum learning (easy samples first)
   - Label smoothing or soft targets
   - Attention transfer from teacher to student
   - Feature-based distillation (intermediate layers, not just final embedding)
4. **Combine best settings**: Take the top-3 individual improvements and try 2-way and 3-way combinations.
5. **Try contrarian experiments**: If all improvements went in one direction (e.g., softer loss), try the opposite extreme to verify the trend isn't local.
6. **Ablation studies**: Remove one component at a time from the current best to understand which parts matter most.
7. **Re-read train.py line by line**: Look for overlooked constants, unused features, or code paths that could be leveraged.

Remember: you are an autonomous AI researcher. Do not just follow a checklist — THINK about what the data is telling you and form hypotheses. The best experiments come from understanding WHY something worked, not just WHAT worked.
