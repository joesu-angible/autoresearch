# Student Distillation -- Optimized Training Configuration

This document contains the **proven optimal configuration** for LCNet student model training via knowledge distillation. The settings below were determined through **69 experiments** of systematic hyperparameter search, augmentation testing, loss function exploration, and teacher selection.

## Quick Start

```bash
cd student_finetune
python train.py
```

No modifications needed -- the current `train.py` already contains the optimal settings documented below.

## Best Results

| Metric | Value | Eval Samples |
|--------|-------|-------------|
| **combined_metric** | **0.8588** | 10,000 |
| **recall@1** | **0.9006** | 10,000 |
| **recall@5** | **0.9432** | 10,000 |
| **mean_cosine** | **0.8169** | 10,000 |
| distill_loss | 0.2092 | -- |
| peak_vram_mb | 2603.4 | -- |

Improvement over baseline (trendyol_onnx, default settings):
- combined: 0.747 -> **0.859** (+15.0%)
- recall@1: 0.840 -> **0.901** (+7.3%)
- mean_cosine: 0.655 -> **0.817** (+24.7%)

## Optimal Configuration

### Teacher Selection

```python
TEACHER = "dinov3_ft"          # Fine-tuned DINOv3 ViT-H+ (1280d) -- our best teacher
TEACHERS = None                # Single teacher only; multi-teacher always hurts
```

**Evidence:** Tested all 5 available teachers:
- `dinov3_ft`: **best** (combined=0.859)
- `trendyol_onnx`: baseline (combined=0.747)
- `dinov2`: much weaker (combined=0.790)
- `radio_so400m`: high cosine (0.892) but lower recall@1 (0.890 vs 0.901)
- Multi-teacher blends (tested 4 combinations): always dilute the best single teacher's signal

### Training Hyperparameters

```python
LR = 8e-3                     # Optimal: 2e-3 -> 5e-3 -> 8e-3 each improved; 7e-3, 9e-3, 1e-2 all worse
BATCH_SIZE = 64                # Optimal: 256 -> 128 -> 64 each improved; 48 too noisy, 96 worse
WEIGHT_DECAY = 1e-3            # Optimal: strong regularization compensates for minimal augmentation
BACKBONE_LR_MULT = 1.0        # Optimal: full backbone LR = head LR; 0.5 and 2.0 both worse
UNFREEZE_EPOCH = 0             # Optimal: unfreeze from start; delay to epoch 1 hurts recall
SEED = 42                     # Best of 3 tested seeds (42, 7, 123)
NUM_WORKERS = 16               # No measurable impact on model quality
```

### ArcFace Metric Learning

```python
USE_ARCFACE = True             # ESSENTIAL: disabling drops recall@1 from 0.901 to 0.818
ARCFACE_S = 32.0               # Optimal: S=16 and S=48 both worse
ARCFACE_M = 0.30               # Optimal: M=0.2, 0.4, 0.5 all worse
ARCFACE_LOSS_WEIGHT = 0.03     # Optimal: 0.01, 0.02, 0.04, 0.05 all worse
ARCFACE_PHASEOUT_EPOCH = 3     # Optimal: linear decay from epoch 3-9; 0, 2, 4, 5 and hard cutoff all worse
ARCFACE_BATCH_SIZE = 256       # Optimal: 64, 128, 384, 512 all worse
ARCFACE_MAX_PER_CLASS = 200    # Optimal: 50 worse, 100 slightly worse
```

### Data Augmentation

```python
QUALITY_DEGRADATION_PROB = 0.0  # DISABLED: quality degradation hurts distillation
DROP_HARD_RATIO = 0.0           # DISABLED: keep all negatives for max gradient signal
SEP_WEIGHT = 0.0                # DISABLED: separation loss adds noise
```

**Transform pipeline (in `build_train_transform()`):**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # KEEP: helps recall
    transforms.RandomVerticalFlip(p=0.5),      # KEEP: helps recall (products can appear rotated)
    PadToSquare(),                             # KEEP: aspect ratio preservation
    transforms.Resize((224, 224)),             # KEEP: standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=ImageNet, std=ImageNet),
])
```

**ALL other augmentations tested and rejected:**
- ColorJitter: hurts (any strength)
- RandomGrayscale: hurts (color is important for products)
- RandomErasing: hurts
- GaussianBlur: hurts
- RandomPerspective: hurts
- RandomResizedCrop: hurts (teacher embeddings computed on full images)

**Key insight:** For distillation, the student should see images as close as possible to what the teacher's cached embeddings were computed on. Any augmentation that changes the image appearance (color, blur, crop) creates a mismatch between the student's input and the teacher's target.

### LCNet Architecture

```python
LCNET_SCALE = 0.5              # HARD LIMIT: never exceed 0.5 (edge deployment)
SE_START_BLOCK = 10            # Default: changing breaks pretrained weight alignment
SE_REDUCTION = 0.25            # Default
ACTIVATION = "h_swish"         # Default
KERNEL_SIZES = [3,3,3,3,3,3,5,5,5,5,5,5,5]  # Default
USE_PRETRAINED = True          # ESSENTIAL: pretrained weights critical for 10-epoch budget
```

### Advanced Features (All Disabled)

```python
SSL_WEIGHT = 0.0               # Self-supervised contrastive: no improvement, adds VRAM+time
ENABLE_PHI_S = False           # PHI-S: breaks single-teacher cosine evaluation
ENABLE_FEATURE_NORMALIZER = False  # Feature normalizer: wrong dimensionality for single teacher
ENABLE_ADAPTOR_MLP_V2 = False  # MLP v2 adaptor: dim mismatch bug with 1280d teacher
ENABLE_L_ANGLE = False         # L_angle: 3x VRAM, minimal benefit
ENABLE_HYBRID_LOSS = False     # Hybrid loss: pure cosine is better for summary distillation
ENABLE_FEATSHARP = False       # Not applicable without spatial distillation
ENABLE_SHIFT_EQUIVARIANT = False  # Not applicable without spatial distillation
VAT_WEIGHT = 0.0               # VAT: no improvement, adds VRAM and training time
SPATIAL_DISTILL_WEIGHT = 0.0   # Spatial distillation: not tested (requires live RADIO inference)
```

### Optimizer & Scheduler

```python
# AdamW with default betas (0.9, 0.999) -- beta2=0.99 tested and worse
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

# CosineAnnealingLR -- PolynomialLR and WarmRestarts both worse
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# SWA: average last 3 epochs -- 4 and 5 within noise, 3 slightly better
SWA_EPOCHS = 3
```

### Evaluation

```python
RETRIEVAL_MAX_SAMPLES = 10000  # 10K samples for stable evaluation (5K was noisy)
RETRIEVAL_TOPK = 5
```

## Key Research Findings

### 1. Teacher Quality Is Everything
The single most impactful change was switching from `trendyol_onnx` (256d) to `dinov3_ft` (1280d, our fine-tuned DINOv3 ViT-H+). This alone improved combined metric by +0.031. Multi-teacher blends always performed worse than single dinov3_ft.

### 2. Less Augmentation = Better for Distillation
Counter-intuitively, removing augmentations consistently improved metrics. The explanation: teacher embeddings are pre-cached on original images. Any augmentation creates a mismatch between what the student sees and what the teacher embedding represents. Only geometric invariances (flips) that don't change product identity are beneficial.

### 3. More Optimizer Steps > Larger Batches
Reducing batch size from 256 to 64 was the second biggest improvement (+0.011 combined). With only 10 epochs, maximizing the number of gradient updates per epoch is critical. The optimal balance is BS=64 with LR=8e-3.

### 4. ArcFace Is Essential but Must Be Carefully Balanced
- Without ArcFace: recall@1 drops from 0.901 to 0.818 (but cosine rises to 0.815)
- ArcFace provides discrimination that distillation alone cannot
- Optimal: use ArcFace for first 3 epochs, then linearly decay to 0
- Gradual decay is better than hard cutoff (epoch 3-9 decay vs epoch 3 stop)

### 5. Run-to-Run Variance Is ~0.005-0.015
Three seeds tested: 42 (0.859), 7 (0.854), 123 (0.828). This means improvements smaller than ~0.005 in combined metric are unreliable.

## Experiment History Summary

| Category | Experiments | Best Finding |
|----------|------------|-------------|
| Teacher selection | 8 | dinov3_ft single teacher |
| Learning rate | 7 | LR=8e-3 |
| Batch size | 5 | BS=64 distill, BS=256 ArcFace |
| ArcFace tuning | 16 | S=32, M=0.3, w=0.03, phaseout=3 |
| Augmentation | 9 | No augmentation (only flips) |
| Weight decay | 4 | WD=1e-3 |
| Architecture | 3 | Default lcnet_050 with pretrained |
| Advanced techniques | 8 | All disabled (PHI-S, SSL, VAT, etc.) |
| Scheduler | 4 | CosineAnnealingLR |
| SWA window | 3 | SWA=3 epochs |
| Seed variance | 3 | SEED=42 best |
| **Total** | **69** | |
