"""Agent-editable V2 training script for DINOv3 LoRA fine-tuning.

Extends v1 (train_dino.py) with:

  Data (always on, no flags — autoresearch principle: more data always):
    - TRAIN_DIR (product_code_dataset/train) — labeled products
    - REID_PRODUCTS_DIR (reid_multiple/products) — labeled products
    - RETAIL_DIR (retail_product_checkout_crop) — labeled retail crops
    - REID_COMMODITY_DIR (reid_multiple/commodity) — 30k flat unlabeled images (SSL only)
    - REID_NEGATIVES_DIR (reid_multiple/negatives) — hard negatives (masked InfoNCE denom only)

  Losses / techniques (flag-gated for ablation):
    - USE_ARCFACE: ArcFace + linear phase-out (student-style)
    - USE_SSL_CONSISTENCY: two-augmented-view embedding alignment
    - USE_NEGATIVES_MASKED_NCE: negatives as InfoNCE denominators only (never positives)
    - USE_STRONG_AUG: ColorJitter + GaussianBlur + RandomErasing
    - USE_BASE_ANCHOR: anchor LoRA embeddings to frozen base DINOv3 (prevents drift)

Target baseline to beat: combined=0.809 (commit 43c0239).

Usage: cd dino_finetune && python train_dino_v2.py
"""

from prepare_dino import (
    load_base_model, get_image_processor, build_dataset,
    extract_cls_embedding, evaluate_dino, save_adapter,
    EPOCHS, EMBEDDING_DIM, ADAPTER_OUTPUT_DIR,
    TRAIN_DIR, VAL_DIR, SKIP_CLASSES,
    REID_PRODUCTS_DIR, RETAIL_DIR,
    set_seed, collate_fn,
)

# -- Production overrides (prepare_dino.py is immutable) --
EPOCHS = 20

# -- New v2 data sources (not in prepare_dino.py) --
REID_COMMODITY_DIR = "/data/training/reid/reid_multiple/commodity"
REID_NEGATIVES_DIR = "/data/training/reid/reid_multiple/negatives"

# Adapter output. Just `best_adapter` — V2 is the only flavor going forward;
# the student trainer reads this same path. Cache invalidation is handled by
# adapter-sha keying in train_v2.py, so overwriting in place is safe.
ADAPTER_OUTPUT_DIR = "dino_finetune/output/best_adapter"
LAST_ADAPTER_DIR = "dino_finetune/output/last_adapter"
CHECKPOINT_PATH = "dino_finetune/output/last_adapter/checkpoint.pt"

import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# EXPERIMENT VARIABLES (agent edits these)
# ============================================================

# -- LoRA --
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# -- Optimization --
def _auto_batch(default_at_24gb: int, vram_gb: float | None = None) -> int:
    """Choose a physical DINO V2 batch size from detected GPU VRAM.

    Gradient checkpointing is enabled by default, so high-VRAM hosts can use a
    large physical batch without OOM. Override `BATCH_SIZE` for manual sweeps.

    Override via env var: `BATCH_SIZE=128 python train_dino_v2.py`.
    Falls back to baseline default on CPU-only systems.
    """
    if vram_gb is None:
        import torch as _torch
        if not _torch.cuda.is_available():
            return default_at_24gb
        vram_gb = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if vram_gb >= 80.0:
        return 128
    if vram_gb >= 48.0:
        return 64
    if vram_gb >= 32.0:
        return 32
    return default_at_24gb


def _auto_num_workers(cpu_count: int | None = None) -> int:
    """Choose DataLoader workers from CPU capacity.

    Image decode + two-view augmentation can bottleneck DINO V2. Scale above the
    legacy 4 workers on large hosts, while capping at 16 to avoid excessive
    process fan-out and memory pressure. Override via `NUM_WORKERS=...`.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 4
    if cpu_count >= 24:
        return 16
    if cpu_count >= 12:
        return 8
    return 4


# Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS is kept near 256 by
# default so high-VRAM hosts use both a large physical batch and accumulation.
# Override either via env vars to tune independently.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE") or _auto_batch(8))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS") or max(1, 256 // BATCH_SIZE))
LR = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.2
TEMPERATURE = 0.20

# -- Data caps --
RETAIL_MAX_PER_CLASS = 100
COMMODITY_MAX_SAMPLES = 20000        # Cap commodity (full 30k overwhelms products)
NEGATIVES_MAX_SAMPLES = 5000         # Cap negatives (full 400k makes batches 75% negative, kills InfoNCE)

# -- Feature flags: Losses / techniques --
USE_ARCFACE = True
ARCFACE_WEIGHT = 0.1
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.3
ARCFACE_PHASEOUT_EPOCH = 5           # Linear decay 0 → weight by this epoch

USE_SSL_CONSISTENCY = False
SSL_WEIGHT = 0.3

USE_NEGATIVES_MASKED_NCE = True      # If False: negatives are ignored entirely
USE_STRONG_AUG = False
USE_BASE_ANCHOR = False
BASE_ANCHOR_WEIGHT = 0.1

# -- Productness CLS branch (issue #5) --
# Mirrors student-side wiring (student_finetune/train_v2.py) but on the DINOv3
# 1280-d CLS embedding. Per project decision 2026-04-25, the teacher must also
# learn productness so its embedding space is product-aware before distillation.
# Target derivation is cleaner here than student: label == NEGATIVE_LABEL → 0.0,
# anything else (real class id or COMMODITY_LABEL) → 1.0. No path-string match.
# OFF by default per algo-lead direction (2026-04-26). Opt-in via CLI
# `--productness` or env `USE_PRODUCTNESS_CLS=1`.
USE_PRODUCTNESS_CLS = (
    os.environ.get("USE_PRODUCTNESS_CLS", "0").lower()
    in ("1", "true", "yes", "on")
)
PRODUCTNESS_CLS_WEIGHT = 0.02
PRODUCTNESS_HEAD_HIDDEN = 256
PRODUCTNESS_LABEL_SMOOTHING_POS = 0.05
PRODUCTNESS_LABEL_SMOOTHING_NEG = 0.02
PRODUCTNESS_FOCAL_GAMMA = 2.0

# -- Training --
SEED = 42
USE_GRADIENT_CHECKPOINTING = False
EVAL_EVERY_N_EPOCHS = 1
MAX_STEPS_PER_EPOCH = 0
MAX_TRAINING_SECONDS = 0

# -- Early stopping --
EARLY_STOP_COSINE_THRESHOLD = 0.95
EARLY_STOP_PATIENCE = 10
EARLY_STOP_RECALL_DROP = 0.15
EARLY_STOP_COLLAPSE_CONSECUTIVE = 3

NUM_WORKERS = int(os.environ.get("NUM_WORKERS") or _auto_num_workers())
DEVICE = "cuda"

# -- Sentinel labels for unlabeled sources --
COMMODITY_LABEL = -1                 # Flat, no class — SSL only
NEGATIVE_LABEL = -2                  # Hard negatives — InfoNCE denom only


# ============================================================
# V2 Multi-source dataset
# ============================================================

class V2MultiSourceDataset(Dataset):
    """Products (labeled) + commodity (unlabeled) + negatives (masked) for DINO v2.

    Label convention:
        >= 0          : product class index (for ArcFace + InfoNCE positives)
        COMMODITY_LABEL (-1): unlabeled commodity (for SSL only)
        NEGATIVE_LABEL  (-2): hard negative (InfoNCE denominator only)

    Returns (image_tensor, label). Augmentation is applied by the transform pipeline.
    The caller should provide two transform instances to do SSL (two-view forward).
    """

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        commodity_root: str | None,
        negatives_root: str | None,
        transform,
        retail_max_per_class: int = 100,
        commodity_max_samples: int = 20000,
        negatives_max_samples: int = 5000,
        skip_classes: set[str] | None = None,
        two_views: bool = False,
    ):
        self.transform = transform
        self.two_views = two_views
        self.samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        # --- 1. Collect primary product classes (union across primary_roots) ---
        ref_class_ids: set[str] = set()
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                logger.warning(f"Primary root missing: {root}")
                continue
            for d in root_path.iterdir():
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                    ref_class_ids.add(d.name)

        # Build ordered product class list, deduping across primary_roots
        product_classes: list[str] = []
        seen: set[str] = set()
        primary_class_dirs: list[Path] = []
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for d in sorted(root_path.iterdir()):
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name in ref_class_ids:
                    primary_class_dirs.append(d)
                    cname = f"primary_{d.name}"
                    if cname not in seen:
                        product_classes.append(cname)
                        seen.add(cname)

        # --- 2. Add retail classes ---
        retail_path = Path(retail_root)
        retail_class_dirs: list[Path] = []
        if retail_path.exists():
            retail_class_dirs = sorted(
                [d for d in retail_path.iterdir()
                 if d.is_dir() and not d.name.startswith((".", "@", "__"))]
            )
            for d in retail_class_dirs:
                product_classes.append(f"retail_{d.name}")

        self.product_classes = product_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(product_classes)}
        self.num_product_classes = len(product_classes)

        # --- 3. Load primary samples (labeled) ---
        primary_count = 0
        for cdir in primary_class_dirs:
            cname = f"primary_{cdir.name}"
            if cname not in self.class_to_idx:
                continue
            cidx = self.class_to_idx[cname]
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                for p in cdir.glob(ext):
                    self.samples.append((str(p), cidx))
                    primary_count += 1

        # --- 4. Load retail samples (labeled, capped per class) ---
        retail_count = 0
        for cdir in retail_class_dirs:
            cname = f"retail_{cdir.name}"
            cidx = self.class_to_idx[cname]
            files: list[Path] = []
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                files.extend(cdir.glob(ext))
            if len(files) > retail_max_per_class:
                files = random.sample(files, retail_max_per_class)
            for p in files:
                self.samples.append((str(p), cidx))
                retail_count += 1

        # --- 5. Commodity (flat, unlabeled) — label = COMMODITY_LABEL ---
        commodity_count = 0
        if commodity_root is not None:
            croot = Path(commodity_root)
            if croot.exists():
                comm_files: list[Path] = []
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                    comm_files.extend(croot.glob(ext))
                if len(comm_files) > commodity_max_samples:
                    comm_files = random.sample(comm_files, commodity_max_samples)
                for p in comm_files:
                    self.samples.append((str(p), COMMODITY_LABEL))
                    commodity_count += 1

        # --- 6. Negatives (labeled pseudo, collapsed to NEGATIVE_LABEL, capped) ---
        negative_count = 0
        if negatives_root is not None:
            nroot = Path(negatives_root)
            if nroot.exists():
                all_negs: list[str] = []
                for d in sorted(nroot.iterdir()):
                    if d.is_dir() and not d.name.startswith((".", "@", "__")):
                        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                            for p in d.glob(ext):
                                all_negs.append(str(p))
                if len(all_negs) > negatives_max_samples:
                    all_negs = random.sample(all_negs, negatives_max_samples)
                for p in all_negs:
                    self.samples.append((p, NEGATIVE_LABEL))
                    negative_count += 1

        logger.info(
            f"V2MultiSourceDataset: "
            f"product_classes={self.num_product_classes} "
            f"(primary={len(seen)}, retail={len(retail_class_dirs)}), "
            f"samples={len(self.samples)} "
            f"(primary={primary_count}, retail={retail_count}, "
            f"commodity={commodity_count}, negatives={negative_count})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")
        if self.two_views:
            # Two independent augmentation passes of the same source image.
            assert self.transform is not None, "two_views requires a transform"
            view_a = self.transform(img)
            view_b = self.transform(img)
            return view_a, view_b, target
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def collate_two_views(batch):
    """Collate (view_a, view_b, label) triplets into stacked tensors."""
    views_a, views_b, labels = zip(*batch)
    return (
        torch.stack(views_a),
        torch.stack(views_b),
        torch.tensor(labels, dtype=torch.long),
    )


def build_v2_train_transform(processor, strong: bool) -> transforms.Compose:
    """Return train transform. `strong=True` enables ColorJitter/Blur/Erasing."""
    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", 518)

    ops: list = [
        transforms.RandomResizedCrop(size, scale=(0.4, 1.0) if strong else (0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if strong:
        ops.extend([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if strong:
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)


def build_v2_train_dataset(processor) -> tuple[V2MultiSourceDataset, int]:
    """Build the v2 training dataset. Excludes val barcodes from product classes."""
    val_dir = Path(VAL_DIR)
    val_barcodes = {
        d.name for d in val_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    } if val_dir.exists() else set()
    skip = SKIP_CLASSES | val_barcodes
    logger.info(f"v2 dataset: skipping {len(skip)} classes ({len(SKIP_CLASSES)} + {len(val_barcodes)} val)")

    transform = build_v2_train_transform(processor, strong=USE_STRONG_AUG)
    dataset = V2MultiSourceDataset(
        primary_roots=[TRAIN_DIR, REID_PRODUCTS_DIR],
        retail_root=RETAIL_DIR,
        commodity_root=REID_COMMODITY_DIR,
        negatives_root=REID_NEGATIVES_DIR,
        transform=transform,
        retail_max_per_class=RETAIL_MAX_PER_CLASS,
        commodity_max_samples=COMMODITY_MAX_SAMPLES,
        negatives_max_samples=NEGATIVES_MAX_SAMPLES,
        skip_classes=skip,
        two_views=USE_SSL_CONSISTENCY,
    )
    return dataset, dataset.num_product_classes


# ============================================================
# Losses
# ============================================================

def masked_info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """InfoNCE with label-aware masking.

    - Samples with label >= 0 are product anchors with positives (same-label in batch).
    - Samples with label < 0 (commodity, negatives) never act as positives for anyone.
    - All non-self samples contribute to the denominator (including negatives as hard distractors).
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature  # [B, B]

    labels_col = labels.unsqueeze(0)
    labels_row = labels.unsqueeze(1)

    # Positive only if both ends have real product labels AND match
    is_real_row = (labels_row >= 0)
    is_real_col = (labels_col >= 0)
    pos_mask = ((labels_row == labels_col) & is_real_row & is_real_col).float()
    pos_mask.fill_diagonal_(0)

    logits_mask = torch.ones_like(sim)
    logits_mask.fill_diagonal_(0)

    exp_logits = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    num_positives = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    valid = num_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return -mean_log_prob[valid].mean()


class ArcFaceHead(torch.nn.Module):
    """ArcFace angular margin classification head (matches v1's ArcFaceHead)."""

    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = torch.nn.Parameter(torch.randn(num_classes, embedding_dim))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        target_logits = torch.cos(theta + self.margin * one_hot)
        return F.cross_entropy(self.scale * target_logits, labels)


def ssl_consistency_loss(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Cosine alignment between two augmented views of same batch."""
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    return (1.0 - (emb_a * emb_b).sum(dim=1)).mean()


def base_anchor_loss(lora_emb: torch.Tensor, base_emb: torch.Tensor) -> torch.Tensor:
    """Keep LoRA embeddings close to the frozen base DINOv3 embeddings."""
    lora_emb = F.normalize(lora_emb, dim=1)
    base_emb = F.normalize(base_emb, dim=1)
    return (1.0 - (lora_emb * base_emb).sum(dim=1)).mean()


# ============================================================
# Productness CLS branch (mirrors student-side wiring)
# ============================================================

class DinoProductnessHead(torch.nn.Module):
    """Auxiliary product-vs-personal-item head on the DINOv3 1280-d CLS embedding.

    Saved alongside the LoRA adapter; not consumed at student-distillation time
    (student only reads cached embeddings). Its purpose is to *shape* the
    teacher's embedding space so personal items separate from products before
    distillation.
    """

    def __init__(self, embedding_dim: int = 1280, hidden: int = PRODUCTNESS_HEAD_HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """[B, 1280] → [B] logits."""
        return self.net(embeddings).squeeze(-1)


def productness_loss_block(
    logits: torch.Tensor,
    y_hard: torch.Tensor,
    eps_pos: float = PRODUCTNESS_LABEL_SMOOTHING_POS,
    eps_neg: float = PRODUCTNESS_LABEL_SMOOTHING_NEG,
    gamma: float = PRODUCTNESS_FOCAL_GAMMA,
) -> torch.Tensor:
    """Asymmetric label smoothing + focal-weighted BCE.

    Identical math to the student-side block in train.py::run_train_epoch — the
    same loss-engineering knobs apply (per ML-engineer review 2026-04-25).
    """
    if eps_pos > 0.0 or eps_neg > 0.0:
        y_target = y_hard * (1.0 - eps_pos) + (1.0 - y_hard) * eps_neg
    else:
        y_target = y_hard
    per_sample = F.binary_cross_entropy_with_logits(logits, y_target, reduction="none")
    if gamma > 0.0:
        with torch.no_grad():
            p = torch.sigmoid(logits)
            p_t = y_hard * p + (1.0 - y_hard) * (1.0 - p)
            focal_w = (1.0 - p_t).pow(gamma)
        return (per_sample * focal_w).mean()
    return per_sample.mean()


def derive_productness_targets(labels: torch.Tensor) -> torch.Tensor:
    """Derive 0/1 productness targets from DINO V2 dataset labels.

    Cleaner than the student's path-string check — DINO already encodes
    "is this a personal item?" via the NEGATIVE_LABEL sentinel.
    """
    return (labels != NEGATIVE_LABEL).float()


# ============================================================
# Optimizer / scheduler (inherit v1 pattern)
# ============================================================

def build_optimizer(model, arcface_head=None, productness_head=None) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if arcface_head is not None:
        params += list(arcface_head.parameters())
    if productness_head is not None:
        params += list(productness_head.parameters())
    return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer, num_training_steps: int):
    warmup_steps = int(num_training_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1.0, float(warmup_steps))
        progress = (step - warmup_steps) / max(1.0, float(num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def arcface_phaseout_weight(epoch: int) -> float:
    """Linear decay: w = ARCFACE_WEIGHT at epoch 0, 0 at ARCFACE_PHASEOUT_EPOCH."""
    if ARCFACE_PHASEOUT_EPOCH <= 0:
        return ARCFACE_WEIGHT
    progress = min(1.0, epoch / float(ARCFACE_PHASEOUT_EPOCH))
    return ARCFACE_WEIGHT * (1.0 - progress)


# ============================================================
# Checkpoint
# ============================================================

def save_last_checkpoint(model, optimizer, scheduler, epoch,
                         best_combined, best_recall,
                         patience_counter, collapse_counter,
                         arcface_head=None):
    os.makedirs(LAST_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(LAST_ADAPTER_DIR)
    state = {
        "epoch": epoch,
        "best_combined": best_combined,
        "best_recall": best_recall,
        "patience_counter": patience_counter,
        "collapse_counter": collapse_counter,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if arcface_head is not None:
        state["arcface_head_state_dict"] = arcface_head.state_dict()
    torch.save(state, CHECKPOINT_PATH)
    logger.info(f"v2 checkpoint saved (epoch {epoch})")


# ============================================================
# Training loop (single epoch)
# ============================================================

def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    processor,
    arcface_head: "ArcFaceHead | None" = None,
    base_model=None,
    productness_head: "DinoProductnessHead | None" = None,
) -> dict:
    """Two-view forward for SSL + main InfoNCE + optional ArcFace + optional base anchor."""
    model.train()
    optimizer.zero_grad()

    arc_w = arcface_phaseout_weight(epoch) if USE_ARCFACE and arcface_head is not None else 0.0

    total_loss = 0.0
    total_nce = 0.0
    total_arc = 0.0
    total_ssl = 0.0
    total_anchor = 0.0
    total_productness = 0.0
    productness_correct = 0
    productness_pos_correct = productness_pos_total = 0
    productness_neg_correct = productness_neg_total = 0
    num_batches = 0

    effective_steps = len(train_loader)
    if MAX_STEPS_PER_EPOCH > 0:
        effective_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    for step, batch in enumerate(train_loader):
        if MAX_STEPS_PER_EPOCH > 0 and step >= MAX_STEPS_PER_EPOCH:
            break

        # Unpack depending on two-view mode
        if USE_SSL_CONSISTENCY:
            images_a, images_b, labels = batch
            images_a = images_a.to(device, dtype=torch.bfloat16, non_blocking=True)
            images_b = images_b.to(device, dtype=torch.bfloat16, non_blocking=True)
        else:
            images_a, labels = batch
            images_a = images_a.to(device, dtype=torch.bfloat16, non_blocking=True)
            images_b = None
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # --- View A (main) ---
            out_a = model(images_a)
            emb_a = extract_cls_embedding(out_a)

            # Main InfoNCE on view A (masked for negatives/commodity)
            if USE_NEGATIVES_MASKED_NCE:
                loss_nce = masked_info_nce_loss(emb_a, labels)
            else:
                # Filter out negatives/commodity entirely
                valid = labels >= 0
                if valid.sum() >= 2:
                    loss_nce = masked_info_nce_loss(emb_a[valid], labels[valid])
                else:
                    loss_nce = torch.tensor(0.0, device=device, requires_grad=True)

            loss = loss_nce

            # --- ArcFace on product samples only ---
            loss_arc_val = 0.0
            if USE_ARCFACE and arcface_head is not None and arc_w > 0:
                product_mask = labels >= 0
                if product_mask.sum() > 0:
                    loss_arc = arcface_head(emb_a[product_mask].float(), labels[product_mask])
                    loss = loss + arc_w * loss_arc
                    loss_arc_val = loss_arc.item()

            # --- SSL consistency (view B): true two-view from independent augmentations ---
            loss_ssl_val = 0.0
            if USE_SSL_CONSISTENCY and images_b is not None:
                out_b = model(images_b)
                emb_b = extract_cls_embedding(out_b)
                loss_ssl = ssl_consistency_loss(emb_a, emb_b)
                loss = loss + SSL_WEIGHT * loss_ssl
                loss_ssl_val = loss_ssl.item()

            # --- Base anchor (optional) ---
            loss_anchor_val = 0.0
            if USE_BASE_ANCHOR and base_model is not None:
                with torch.no_grad():
                    base_out = base_model(images_a)
                    base_emb = extract_cls_embedding(base_out)
                loss_anchor = base_anchor_loss(emb_a, base_emb)
                loss = loss + BASE_ANCHOR_WEIGHT * loss_anchor
                loss_anchor_val = loss_anchor.item()

            # --- Productness CLS (issue #5) on view A's CLS embedding ---
            loss_productness_val = 0.0
            if productness_head is not None and PRODUCTNESS_CLS_WEIGHT > 0:
                y_hard = derive_productness_targets(labels).to(emb_a.dtype)
                productness_logits = productness_head(emb_a.float()).to(emb_a.dtype)
                loss_productness = productness_loss_block(
                    productness_logits.float(),
                    y_hard.float(),
                    eps_pos=PRODUCTNESS_LABEL_SMOOTHING_POS,
                    eps_neg=PRODUCTNESS_LABEL_SMOOTHING_NEG,
                    gamma=PRODUCTNESS_FOCAL_GAMMA,
                )
                loss = loss + PRODUCTNESS_CLS_WEIGHT * loss_productness
                loss_productness_val = float(loss_productness.item())
                with torch.no_grad():
                    pred = (productness_logits > 0).float()
                    pos_mask = y_hard > 0.5
                    neg_mask = ~pos_mask
                    productness_correct += int((pred == y_hard).sum().item())
                    productness_pos_total += int(pos_mask.sum().item())
                    productness_neg_total += int(neg_mask.sum().item())
                    if pos_mask.any():
                        productness_pos_correct += int((pred[pos_mask] == 1.0).sum().item())
                    if neg_mask.any():
                        productness_neg_correct += int((pred[neg_mask] == 0.0).sum().item())

            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_nce += loss_nce.item()
        total_arc += loss_arc_val
        total_ssl += loss_ssl_val
        total_anchor += loss_anchor_val
        total_productness += loss_productness_val
        num_batches += 1

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} step {step+1}/{effective_steps}: "
                f"loss={total_loss/num_batches:.4f} "
                f"nce={total_nce/num_batches:.4f} "
                f"arc={total_arc/num_batches:.4f} (w={arc_w:.4f}) "
                f"ssl={total_ssl/num_batches:.4f} "
                f"anchor={total_anchor/num_batches:.4f} "
                f"lr={lr_now:.2e}"
            )

    if num_batches % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if epoch == 1:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak VRAM after epoch 1: {peak_vram_mb:.0f} MB")

    _prod_n = productness_pos_total + productness_neg_total
    return {
        "loss": total_loss / max(num_batches, 1),
        "nce": total_nce / max(num_batches, 1),
        "arc": total_arc / max(num_batches, 1),
        "ssl": total_ssl / max(num_batches, 1),
        "anchor": total_anchor / max(num_batches, 1),
        "arc_weight": arc_w,
        "productness_loss": total_productness / max(num_batches, 1) if productness_head is not None else 0.0,
        "productness_acc": (productness_correct / _prod_n) if _prod_n > 0 else 0.0,
        "productness_pos_acc": (productness_pos_correct / productness_pos_total) if productness_pos_total > 0 else 0.0,
        "productness_neg_acc": (productness_neg_correct / productness_neg_total) if productness_neg_total > 0 else 0.0,
        "productness_n": _prod_n,
    }


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info("DINOv3 ViT-H+ LoRA Fine-tuning — V2")
    logger.info("=" * 60)
    logger.info(
        f"Flags: ArcFace={USE_ARCFACE}(w={ARCFACE_WEIGHT},phaseout={ARCFACE_PHASEOUT_EPOCH}) "
        f"SSL={USE_SSL_CONSISTENCY}(w={SSL_WEIGHT}) "
        f"MaskedNCE={USE_NEGATIVES_MASKED_NCE} "
        f"StrongAug={USE_STRONG_AUG} "
        f"BaseAnchor={USE_BASE_ANCHOR}(w={BASE_ANCHOR_WEIGHT})"
    )
    logger.info(
        f"Throughput config: BATCH_SIZE={BATCH_SIZE} "
        f"GRADIENT_ACCUMULATION_STEPS={GRADIENT_ACCUMULATION_STEPS} "
        f"effective_batch={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} "
        f"NUM_WORKERS={NUM_WORKERS}"
    )

    # -- Load base model --
    base_model = load_base_model(DEVICE)

    # -- Resume or fresh LoRA --
    has_adapter = os.path.exists(os.path.join(LAST_ADAPTER_DIR, "adapter_model.safetensors"))
    has_checkpoint = os.path.exists(CHECKPOINT_PATH)

    if has_adapter:
        from peft import PeftModel
        logger.info(f"Loading saved adapter from {LAST_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, LAST_ADAPTER_DIR, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias="none",
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # -- Separate frozen base model for BASE_ANCHOR (optional) --
    anchor_base = None
    if USE_BASE_ANCHOR:
        logger.info("Loading SECOND frozen base DINOv3 for base anchor loss")
        anchor_base = load_base_model(DEVICE)

    # -- Data --
    processor = get_image_processor()
    train_dataset, num_product_classes = build_v2_train_dataset(processor)
    val_dataset, _ = build_dataset(VAL_DIR, processor, split="val")

    train_collate = collate_two_views if USE_SSL_CONSISTENCY else collate_fn
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        collate_fn=train_collate, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        collate_fn=collate_fn, **loader_kwargs,
    )

    # -- ArcFace head --
    arcface_head = None
    if USE_ARCFACE:
        arcface_head = ArcFaceHead(
            EMBEDDING_DIM, num_product_classes,
            ARCFACE_SCALE, ARCFACE_MARGIN,
        ).to(DEVICE)
        logger.info(f"ArcFace head: {num_product_classes} classes, s={ARCFACE_SCALE}, m={ARCFACE_MARGIN}")

    # -- Productness CLS head (issue #5; teacher-side mirror of student) --
    productness_head = None
    if USE_PRODUCTNESS_CLS:
        productness_head = DinoProductnessHead(
            embedding_dim=EMBEDDING_DIM,
            hidden=PRODUCTNESS_HEAD_HIDDEN,
        ).to(DEVICE)
        logger.info(
            f"Productness CLS head: weight={PRODUCTNESS_CLS_WEIGHT}, "
            f"hidden={PRODUCTNESS_HEAD_HIDDEN}, "
            f"smoothing=({PRODUCTNESS_LABEL_SMOOTHING_POS},{PRODUCTNESS_LABEL_SMOOTHING_NEG}), "
            f"focal_gamma={PRODUCTNESS_FOCAL_GAMMA}"
        )

    optimizer = build_optimizer(model, arcface_head=arcface_head, productness_head=productness_head)
    steps_per_epoch = (
        min(len(train_loader), MAX_STEPS_PER_EPOCH) if MAX_STEPS_PER_EPOCH > 0
        else len(train_loader)
    )
    num_training_steps = math.ceil(steps_per_epoch / GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    scheduler = build_scheduler(optimizer, num_training_steps)

    # -- Resume state --
    best_combined = -1.0
    best_recall = -1.0
    patience_counter = 0
    collapse_counter = 0
    start_epoch = 1

    if has_adapter and has_checkpoint:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        best_combined = ckpt["best_combined"]
        best_recall = ckpt["best_recall"]
        patience_counter = ckpt["patience_counter"]
        collapse_counter = ckpt["collapse_counter"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if arcface_head is not None and "arcface_head_state_dict" in ckpt:
            arcface_head.load_state_dict(ckpt["arcface_head_state_dict"])
        logger.info(f"Resume from epoch {start_epoch}, best_combined={best_combined:.4f}")
    elif has_adapter:
        logger.info("Warm start: adapter loaded, fresh optimizer")

    # -- Train --
    start_time = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        elapsed = time.time() - start_time
        if MAX_TRAINING_SECONDS > 0 and elapsed >= MAX_TRAINING_SECONDS:
            logger.info(f"Time budget exhausted at epoch {epoch-1}")
            break

        epoch_start = time.time()
        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch,
            processor=processor,
            arcface_head=arcface_head,
            base_model=anchor_base,
            productness_head=productness_head,
        )
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{EPOCHS}: loss={stats['loss']:.4f} "
            f"nce={stats['nce']:.4f} arc={stats['arc']:.4f}(w={stats['arc_weight']:.3f}) "
            f"ssl={stats['ssl']:.4f} anchor={stats['anchor']:.4f} "
            f"time={epoch_time:.1f}s"
        )
        if productness_head is not None:
            logger.info(
                f"  Productness: loss={stats['productness_loss']:.4f} "
                f"acc={stats['productness_acc']:.4f} "
                f"pos_acc={stats['productness_pos_acc']:.4f} "
                f"neg_acc={stats['productness_neg_acc']:.4f} "
                f"(n={stats['productness_n']})"
            )

        # -- Eval + early stop --
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            metrics = evaluate_dino(model, val_loader, DEVICE)
            current_recall = metrics["recall@1"]
            current_cosine = metrics["mean_cosine"]

            # Atomic partial-state dump for tournament timeout recovery (T1).
            # Mirrors the student-side schema; no productness val split on
            # DINO yet, so train-side stats are reported under both keys.
            import json as _json
            progress_payload = {
                "status": "in_progress",
                "version": "v2",
                "is_partial": True,
                "epochs_completed": int(epoch),
                "max_epochs": int(EPOCHS),
                "combined_metric": float(metrics["combined"]),
                "best_combined": float(max(best_combined, metrics["combined"])),
                "recall_at_1": float(current_recall),
                "recall_at_5": float(metrics.get("recall@5", 0.0)),
                "mean_cosine": float(current_cosine),
                "productness_loss": float(stats.get("productness_loss", 0.0)),
                "productness_acc": float(stats.get("productness_acc", 0.0)),
                "productness_pos_acc": float(stats.get("productness_pos_acc", 0.0)),
                "productness_neg_acc": float(stats.get("productness_neg_acc", 0.0)),
            }
            progress_path = Path(ADAPTER_OUTPUT_DIR).parent / "metrics_progress_v2.json"
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(progress_payload, indent=2))
            tmp.replace(progress_path)

            is_collapsed = current_cosine > EARLY_STOP_COSINE_THRESHOLD
            if is_collapsed:
                collapse_counter += 1
                logger.warning(
                    f"Cosine collapse (mean_cosine={current_cosine:.4f}), "
                    f"consecutive={collapse_counter}/{EARLY_STOP_COLLAPSE_CONSECUTIVE}"
                )
                if collapse_counter >= EARLY_STOP_COLLAPSE_CONSECUTIVE:
                    logger.warning("EARLY STOP: cosine collapse")
                    break
            else:
                collapse_counter = 0
                improved = metrics["combined"] > best_combined
                if improved:
                    best_combined = metrics["combined"]
                    save_adapter(model, output_dir=ADAPTER_OUTPUT_DIR)
                    if productness_head is not None:
                        torch.save(
                            productness_head.state_dict(),
                            Path(ADAPTER_OUTPUT_DIR) / "productness_head.pt",
                        )
                    logger.info(f"New best combined={best_combined:.4f}, adapter saved")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        logger.warning(f"EARLY STOP: patience {EARLY_STOP_PATIENCE}")
                        break

            if current_recall > best_recall:
                best_recall = current_recall
            if best_recall > 0 and (best_recall - current_recall) > EARLY_STOP_RECALL_DROP:
                logger.warning(f"EARLY STOP: recall drop {best_recall - current_recall:.4f}")
                break

            save_last_checkpoint(
                model, optimizer, scheduler, epoch,
                best_combined, best_recall, patience_counter, collapse_counter,
                arcface_head=arcface_head,
            )

    # -- Final eval on best adapter --
    total_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    logger.info("Loading best v2 adapter for final eval...")
    from prepare_dino import load_finetuned_model
    best_model = load_finetuned_model(ADAPTER_OUTPUT_DIR, DEVICE)
    final = evaluate_dino(best_model, val_loader, DEVICE)

    logger.info("=" * 60)
    logger.info(
        f"RESULT V2: recall@1={final['recall@1']:.4f} "
        f"mean_cosine={final['mean_cosine']:.4f} "
        f"combined={final['combined']:.4f} "
        f"peak_vram_mb={int(peak_vram_mb)}"
    )
    print(f"METRIC: {final['combined']:.6f}")
    logger.info(f"Total training time: {total_time:.1f}s")

    # Dump structured metrics so the tournament adapter can parse them.
    import json
    metrics_out = {
        "status": "success",
        "version": "v2",
        "combined_metric": float(final["combined"]),
        "best_combined": float(best_combined),
        "recall_at_1": float(final["recall@1"]),
        "recall_at_5": float(final.get("recall@5", 0.0)),
        "mean_cosine": float(final["mean_cosine"]),
        "peak_vram_mb": float(peak_vram_mb),
        "elapsed_seconds": round(total_time, 1),
        "epochs_trained": int(epoch),
    }
    metrics_path = Path(ADAPTER_OUTPUT_DIR).parent / "metrics_final_v2.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    logger.info(f"Wrote {metrics_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V2 DINOv3 LoRA fine-tuning")
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help=f"Override module-level EPOCHS (default {EPOCHS})",
    )
    parser.add_argument(
        "--productness",
        action=argparse.BooleanOptionalAction, default=None,
        help="Toggle productness CLS branch (use --productness or --no-productness). "
             "Default reads USE_PRODUCTNESS_CLS env var (default OFF). "
             "CLI flag overrides env.",
    )
    args = parser.parse_args()
    if args.max_epochs is not None:
        EPOCHS = args.max_epochs  # noqa: F811 — module-level rebind for tournament use
        logger.info(f"EPOCHS overridden via CLI: {EPOCHS}")
    if args.productness is not None:
        USE_PRODUCTNESS_CLS = args.productness  # noqa: F811
        logger.info(f"USE_PRODUCTNESS_CLS overridden via CLI: {USE_PRODUCTNESS_CLS}")

    try:
        main()
    except torch.cuda.OutOfMemoryError:
        logger.error("CRASH: OOM")
        print("CRASH: OOM")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CRASH: OOM")
            print("CRASH: OOM")
        else:
            raise
