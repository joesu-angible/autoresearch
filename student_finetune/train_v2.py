"""V2 training script for LCNet student distillation.

Extends train_final.py (optimal config from 69 experiments) with:

  Data (always on — autoresearch principle: more data always):
    - TRAIN_DIR (product_code_dataset/train) — 1068 labeled classes, 48322 images
        This was defined in prepare.py but NEVER referenced in v1 training. Pure waste fix.
    - REID_COMMODITY_FLAT (reid_multiple/commodity) — 30000 flat unlabeled jpg
        v1 passed it to CombinedDistillDataset but `d.is_dir()` filter silently dropped all of it.
        v2 handles flat-file layout correctly: each image gets a pseudo-unlabeled label
        and flows through distillation only (no ArcFace).
    - REID_PRODUCTS, ARCFACE_DIR, REID_NEGATIVES: unchanged from v1

  Techniques:
    - USE_STRONG_AUG: ColorJitter/GaussianBlur/RandomErasing
    - ArcFace + phase-out: inherited from train_final.py (already optimal)
    - Negatives as blacklist: inherited from v1

  Design note: SSL consistency NOT included in this v2 because student's distillation
  already teaches augmentation invariance implicitly (teacher produces ONE target
  per image, different augs all pulled to same target). SSL would be redundant here.
  DINO v2 adds SSL because DINO uses pure InfoNCE without teacher regression.

Target baseline to beat: combined=0.8588 (commit 87766d9).

Usage:
  python train_v2.py                          # Train from scratch
  python train_v2.py --resume                 # Resume from last.pt
  python train_v2.py --max-epochs 30          # Custom epoch count
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prepare import (
    CombinedArcFaceDataset,
    collate_distill, collate_arcface,
    PadToSquare,
    load_teacher_embeddings,
    TEACHER_REGISTRY, init_teachers, build_all_teacher_caches,
    run_retrieval_eval, compute_combined_metric,
    build_val_transform,
    build_arcface_dataset, build_val_dataset,
    EMBEDDING_DIM, IMAGE_SIZE,
    TRAIN_DIR, VAL_DIR, ARCFACE_DIR,
    REID_PRODUCTS, REID_COMMODITY, REID_NEGATIVES,
    SKIP_CLASSES,
    set_seed,
)

from train import (
    LCNet, load_pretrained_lcnet, ArcMarginProduct,
    RandomQualityDegradation,
    run_train_epoch, EpochStats, make_divisible,
)


# ============================================================
# V2 CONFIGURATION
# ============================================================
MODEL_NAME = "hf-hub:timm/lcnet_050.ra2_in1k"
BATCH_SIZE = 64
ARCFACE_BATCH_SIZE = 256
LR = 8e-3
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 16
SEED = 42
DEVICE = "cuda"

# Teacher
TEACHER = "dinov3_ft"

# ArcFace (inherited from train_final.py optimal)
USE_ARCFACE = True
ARCFACE_S = 32.0
ARCFACE_M = 0.30
ARCFACE_LOSS_WEIGHT = 0.03
ARCFACE_PHASEOUT_EPOCH = 3
ARCFACE_MAX_PER_CLASS = 200

# Architecture (inherited)
LCNET_SCALE = 0.5
SE_START_BLOCK = 10
SE_REDUCTION = 0.25
ACTIVATION = "h_swish"
KERNEL_SIZES = [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5]
USE_PRETRAINED = True

# Training
BACKBONE_LR_MULT = 1.0
DROP_HARD_RATIO = 0.0
QUALITY_DEGRADATION_PROB = 0.0

# Evaluation
RETRIEVAL_MAX_SAMPLES = 10000
RETRIEVAL_TOPK = 5

# V2 data caps
COMMODITY_MAX_SAMPLES = 20000       # Cap commodity (full 30k overwhelms products)

# V2 feature flags
USE_STRONG_AUG = True

# V2 output (isolated from v1)
OUTPUT_DIR = "/data/training/reid/workspace/output/distill_final_lcnet050_v2"


# ============================================================
# V2 Transform
# ============================================================

def build_v2_train_transform(image_size: int, strong: bool = True) -> transforms.Compose:
    """Build training transform. `strong=True` enables ColorJitter/Blur/Erasing."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    ops: list = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        PadToSquare(),
    ]
    if strong:
        ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    ops.append(transforms.Resize((image_size, image_size)))
    if strong:
        ops.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if strong:
        ops.append(transforms.RandomErasing(p=0.2))
    return transforms.Compose(ops)


# ============================================================
# V2 Distillation Dataset — adds TRAIN_DIR + commodity flat
# ============================================================

class V2CombinedDistillDataset(Dataset):
    """v2: Products (labeled) + commodity (flat unlabeled) + negatives (blacklist).

    Extends the v1 CombinedDistillDataset by:
      1. Accepting TRAIN_DIR as another primary root (v1 never used it)
      2. Handling commodity as a FLAT unlabeled source (v1 silently dropped it)
         Each commodity image gets its own pseudo-class "unlabeled_commodity"
         which ArcFace excludes (handled by skip_extra_classes in build_v2_arcface_dataset).

    Return format: (img, label, path) — matches v1 for run_train_epoch compatibility.

    Replacement ratios (fractions of each batch):
      blacklist_ratio: replace with hard negative
      commodity_ratio: replace with unlabeled commodity (distill-only signal)
      retail_ratio: replace with retail crop (labeled)
      else: use primary index
    """

    COMMODITY_PSEUDOCLASS = "unlabeled_commodity"

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        commodity_flat_root: str | None,
        blacklist_root: str | None,
        transform: Callable | None = None,
        retail_ratio: float = 0.25,
        commodity_ratio: float = 0.15,
        blacklist_ratio: float = 0.10,
        commodity_max_samples: int = 20000,
        skip_classes: set[str] | None = None,
        quality_degradation: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.retail_ratio = retail_ratio
        self.commodity_ratio = commodity_ratio
        self.blacklist_ratio = blacklist_ratio
        self.quality_degradation = quality_degradation

        self.samples: list[tuple[str, int]] = []
        self.retail_samples: list[tuple[str, int]] = []
        self.commodity_samples: list[tuple[str, int]] = []
        self.blacklist_samples: list[tuple[str, int]] = []

        _skip = skip_classes or set()

        # --- 1. Primary class union (TRAIN_DIR + REID_PRODUCTS) ---
        all_primary_classes: set[str] = set()
        for root in primary_roots:
            rp = Path(root)
            if rp.exists():
                for d in rp.iterdir():
                    if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                        all_primary_classes.add(d.name)

        # --- 2. Retail classes (prefixed) ---
        retail_path = Path(retail_root)
        retail_class_dirs: list[Path] = []
        if retail_path.exists():
            retail_class_dirs = sorted(
                [d for d in retail_path.iterdir()
                 if d.is_dir() and not d.name.startswith((".", "@", "__"))]
            )
        retail_classes = [f"retail_{d.name}" for d in retail_class_dirs]

        # --- 3. Blacklist classes (prefixed) ---
        bl_class_dirs: list[Path] = []
        if blacklist_root:
            bp = Path(blacklist_root)
            if bp.exists():
                bl_class_dirs = sorted(
                    [d for d in bp.iterdir()
                     if d.is_dir() and not d.name.startswith((".", "@", "__"))]
                )
        bl_classes = [f"bl_{d.name}" for d in bl_class_dirs]

        # --- 4. Assemble class list ---
        #   Ordering: primaries, retail, blacklist, commodity (as last class)
        self.classes: list[str] = (
            sorted(all_primary_classes) + retail_classes + bl_classes
        )
        if commodity_flat_root is not None and Path(commodity_flat_root).exists():
            self.classes.append(self.COMMODITY_PSEUDOCLASS)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.blacklist_class_indices: set[int] = {self.class_to_idx[c] for c in bl_classes}

        # --- 5. Load primary samples from ALL primary_roots ---
        primary_count = 0
        for root in primary_roots:
            rp = Path(root)
            if not rp.exists():
                logger.warning(f"Primary root missing: {root}")
                continue
            for cdir in sorted(rp.iterdir()):
                if not (cdir.is_dir() and not cdir.name.startswith((".", "@", "__"))
                        and cdir.name not in _skip):
                    continue
                cidx = self.class_to_idx[cdir.name]
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                    for p in cdir.glob(ext):
                        self.samples.append((str(p), cidx))
                        primary_count += 1

        # --- 6. Load retail samples (for random replacement) ---
        retail_count = 0
        for cdir in retail_class_dirs:
            cname = f"retail_{cdir.name}"
            cidx = self.class_to_idx[cname]
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                for p in cdir.glob(ext):
                    self.retail_samples.append((str(p), cidx))
                    retail_count += 1

        # --- 7. Load blacklist samples (capped) ---
        max_bl = 50_000
        all_bl: list[tuple[str, int]] = []
        for cdir in bl_class_dirs:
            cname = f"bl_{cdir.name}"
            cidx = self.class_to_idx[cname]
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                for p in cdir.glob(ext):
                    all_bl.append((str(p), cidx))
        if len(all_bl) > max_bl:
            all_bl = random.sample(all_bl, max_bl)
        self.blacklist_samples = all_bl
        bl_count = len(all_bl)

        # --- 8. Load commodity samples (flat, unlabeled pseudo-class) ---
        commodity_count = 0
        if commodity_flat_root is not None:
            crp = Path(commodity_flat_root)
            if crp.exists():
                cidx = self.class_to_idx[self.COMMODITY_PSEUDOCLASS]
                files: list[Path] = []
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                    files.extend(crp.glob(ext))
                if len(files) > commodity_max_samples:
                    files = random.sample(files, commodity_max_samples)
                for p in files:
                    self.commodity_samples.append((str(p), cidx))
                    commodity_count += 1

        logger.info(
            f"V2CombinedDistillDataset: {len(self.classes)} classes "
            f"(primary={len(all_primary_classes)}, retail={len(retail_class_dirs)}, "
            f"blacklist={len(bl_class_dirs)}, commodity={1 if commodity_count else 0}), "
            f"samples: primary={primary_count}, retail={retail_count}, "
            f"blacklist={bl_count}, commodity={commodity_count} "
            f"(ratios: retail={retail_ratio} commodity={commodity_ratio} bl={blacklist_ratio})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        r = random.random()
        acc = 0.0
        # Priority: blacklist > commodity > retail > primary
        if self.blacklist_samples and r < (acc := acc + self.blacklist_ratio):
            path, target = random.choice(self.blacklist_samples)
        elif self.commodity_samples and r < (acc := acc + self.commodity_ratio):
            path, target = random.choice(self.commodity_samples)
        elif self.retail_samples and r < (acc := acc + self.retail_ratio):
            path, target = random.choice(self.retail_samples)
        else:
            path, target = self.samples[index]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")

        if self.quality_degradation is not None:
            img = self.quality_degradation(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


# ============================================================
# V2 Dataset builders (local — do not modify prepare.py)
# ============================================================

def build_v2_distill_dataset(transform, quality_degradation) -> V2CombinedDistillDataset:
    """V2 distillation dataset: adds TRAIN_DIR + commodity flat-file handling."""
    return V2CombinedDistillDataset(
        primary_roots=[TRAIN_DIR, REID_PRODUCTS],
        retail_root=ARCFACE_DIR,
        commodity_flat_root=REID_COMMODITY,
        blacklist_root=REID_NEGATIVES,
        transform=transform,
        retail_ratio=0.25,
        commodity_ratio=0.15,
        blacklist_ratio=0.10,
        commodity_max_samples=COMMODITY_MAX_SAMPLES,
        skip_classes=SKIP_CLASSES,
        quality_degradation=quality_degradation,
    )


def build_v2_arcface_dataset(transform, quality_degradation, max_per_class):
    """V2 ArcFace dataset: adds TRAIN_DIR as another primary root.

    Uses CombinedArcFaceDataset directly. It already filters by class membership
    and excludes val barcodes (via skip_classes), so we just extend primary_roots.
    """
    val_dir = Path(VAL_DIR)
    val_barcodes = {
        d.name for d in val_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    } if val_dir.exists() else set()
    arc_skip = SKIP_CLASSES | val_barcodes
    logger.info(f"V2 ArcFace: skipping {len(arc_skip)} classes (1 empty + {len(val_barcodes)} val)")
    dataset = CombinedArcFaceDataset(
        primary_roots=[TRAIN_DIR, REID_PRODUCTS],
        retail_root=ARCFACE_DIR,
        transform=transform,
        retail_max_per_class=max_per_class,
        skip_classes=arc_skip,
        quality_degradation=quality_degradation,
        skip_degradation_paths=[],
    )
    return dataset, len(dataset.classes)


# ============================================================
# CHECKPOINT (inherited from train_final.py)
# ============================================================

def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, arc_margin,
    epoch, best_combined, combined, recall_at_1, mean_cosine,
    swa_state, swa_count, no_improve_count,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_combined": best_combined,
        "combined_metric": combined,
        "recall_at_1": recall_at_1,
        "mean_cosine": mean_cosine,
        "swa_state": swa_state,
        "swa_count": swa_count,
        "no_improve_count": no_improve_count,
    }
    if arc_margin is not None:
        ckpt["arc_margin_state_dict"] = arc_margin.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, arc_margin, device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    if arc_margin is not None and "arc_margin_state_dict" in ckpt:
        arc_margin.load_state_dict(ckpt["arc_margin_state_dict"])
    return ckpt


# ============================================================
# MAIN
# ============================================================

def main(max_epochs: int, patience: int, swa_epochs: int, resume: bool) -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== V2 Training: max_epochs={max_epochs}, patience={patience}, swa_epochs={swa_epochs} ===")
    logger.info(f"V2 flags: USE_STRONG_AUG={USE_STRONG_AUG}, COMMODITY_MAX={COMMODITY_MAX_SAMPLES}")

    # --- Data ---
    train_transform = build_v2_train_transform(IMAGE_SIZE, strong=USE_STRONG_AUG)
    quality_degradation = RandomQualityDegradation(prob=QUALITY_DEGRADATION_PROB)
    distill_dataset = build_v2_distill_dataset(train_transform, quality_degradation)

    arcface_dataset, num_arcface_classes = None, 0
    if USE_ARCFACE:
        arcface_dataset, num_arcface_classes = build_v2_arcface_dataset(
            train_transform, quality_degradation, max_per_class=ARCFACE_MAX_PER_CLASS
        )

    val_dataset = build_val_dataset(MODEL_NAME, IMAGE_SIZE)

    distill_loader = DataLoader(
        distill_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=collate_distill,
    )
    arcface_loader = None
    if arcface_dataset is not None:
        arcface_loader = DataLoader(
            arcface_dataset, batch_size=ARCFACE_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            collate_fn=collate_arcface,
        )

    # --- Teacher ---
    teacher_names = [TEACHER]
    teacher_weights = {TEACHER: 1.0}
    teacher_dims = {TEACHER: TEACHER_REGISTRY[TEACHER]["embedding_dim"]}

    all_image_paths = [s[0] for s in distill_dataset.samples]
    if distill_dataset.retail_samples:
        all_image_paths += [s[0] for s in distill_dataset.retail_samples]
    if distill_dataset.commodity_samples:
        all_image_paths += [s[0] for s in distill_dataset.commodity_samples]
    if distill_dataset.blacklist_samples:
        all_image_paths += [s[0] for s in distill_dataset.blacklist_samples]
    all_image_paths = list(set(all_image_paths))
    logger.info(f"Total unique image paths for teacher cache: {len(all_image_paths)}")

    build_all_teacher_caches(teacher_names, all_image_paths, device=str(device))
    teachers_dict = init_teachers(teacher_names, device=str(device))
    logger.info(f"Teacher: {TEACHER}, dim={teacher_dims[TEACHER]}")

    # --- Model ---
    model = LCNet(
        scale=LCNET_SCALE, se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION, activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES, embedding_dim=EMBEDDING_DIM,
        device=str(device), teacher_dims=teacher_dims,
    ).to(device)

    if USE_PRETRAINED:
        load_pretrained_lcnet(model, LCNET_SCALE)
    model.unfreeze_last_stage()

    # --- ArcFace ---
    arc_margin = None
    if USE_ARCFACE and arcface_dataset is not None:
        arc_margin = ArcMarginProduct(
            in_features=EMBEDDING_DIM, out_features=num_arcface_classes,
            s=ARCFACE_S, m=ARCFACE_M,
        ).to(device)
        logger.info(f"ArcFace: {num_arcface_classes} classes, s={ARCFACE_S}, m={ARCFACE_M}")

    # --- Optimizer ---
    backbone_params = [p for p in list(model.conv_stem.parameters()) +
                       list(model.bn1.parameters()) +
                       list(model.blocks.parameters()) if p.requires_grad]
    if hasattr(model, 'proj_heads') and model.proj_heads is not None:
        proj_params = []
        for ph in model.proj_heads.values():
            proj_params += list(ph.parameters())
    else:
        proj_params = list(model.proj.parameters())
    head_params = proj_params + list(model.conv_head.parameters()) + list(model.head_act.parameters())
    if arc_margin is not None:
        head_params += list(arc_margin.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR},
            {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # --- Resume ---
    start_epoch = 0
    best_combined = 0.0
    no_improve_count = 0
    swa_state: dict | None = None
    swa_count = 0
    swa_start = max_epochs - swa_epochs

    last_pt = out_dir / "last.pt"
    best_pt = out_dir / "best.pt"

    if resume and last_pt.exists():
        logger.info(f"Resuming from {last_pt}")
        ckpt = load_checkpoint(last_pt, model, optimizer, scheduler, scaler, arc_margin, device)
        start_epoch = ckpt["epoch"] + 1
        best_combined = ckpt["best_combined"]
        no_improve_count = ckpt.get("no_improve_count", 0)
        swa_state = ckpt.get("swa_state")
        swa_count = ckpt.get("swa_count", 0)
        logger.info(f"Resumed at epoch {start_epoch}, best_combined={best_combined:.6f}")

    # --- Train ---
    wl_centroid_ema: dict = {"centroid": None}
    t_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        # ArcFace phase-out
        if ARCFACE_PHASEOUT_EPOCH > 0 and epoch >= ARCFACE_PHASEOUT_EPOCH:
            remaining = max_epochs - ARCFACE_PHASEOUT_EPOCH
            progress = (epoch - ARCFACE_PHASEOUT_EPOCH) / max(remaining, 1)
            effective_arc_weight = ARCFACE_LOSS_WEIGHT * (1.0 - progress)
        else:
            effective_arc_weight = ARCFACE_LOSS_WEIGHT

        stats = run_train_epoch(
            model=model,
            distill_loader=distill_loader,
            arcface_loader=arcface_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            teachers=teachers_dict,
            teacher_weights=teacher_weights,
            device=device,
            amp=(device.type == "cuda"),
            arc_margin=arc_margin,
            arc_loss_weight=effective_arc_weight,
            drop_hard_ratio=DROP_HARD_RATIO,
            blacklist_class_indices=distill_dataset.blacklist_class_indices,
            wl_centroid_ema=wl_centroid_ema,
            backbone_unfrozen=True,
            save_first_batch_path=out_dir if epoch == 0 else None,
        )
        epoch_time = time.time() - t0

        arc_str = f" arc_w={effective_arc_weight:.4f}" if effective_arc_weight != ARCFACE_LOSS_WEIGHT else ""
        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"loss={stats.loss:.4f} distill={stats.distill_loss:.4f} "
            f"arc={stats.arc_loss:.4f} cosine={stats.mean_cosine:.4f}{arc_str} | {epoch_time:.1f}s"
        )

        # SWA
        if epoch >= swa_start:
            sd = {k: v.clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = sd
            else:
                for k in swa_state:
                    swa_state[k] += sd[k]
            swa_count += 1
            logger.info(f"  SWA: accumulated ({swa_count}/{swa_epochs})")

        # Eval
        recall_at_1 = recall_at_5 = 0.0
        mean_cos = stats.mean_cosine
        if val_dataset is not None:
            metrics = run_retrieval_eval(
                model=model, dataset=val_dataset, device=device,
                amp=(device.type == "cuda"), max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK, seed=SEED, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = metrics["recall@1"]
            recall_at_5 = metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            logger.info(f"  Retrieval: recall@1={recall_at_1:.4f} recall@5={recall_at_5:.4f}")

        combined = compute_combined_metric(recall_at_1, mean_cos)
        logger.info(f"  Combined: {combined:.6f} (best={best_combined:.6f})")

        # Save last + best
        save_checkpoint(
            last_pt, model, optimizer, scheduler, scaler, arc_margin,
            epoch, best_combined, combined, recall_at_1, mean_cos,
            swa_state, swa_count, no_improve_count,
        )

        if combined > best_combined:
            best_combined = combined
            no_improve_count = 0
            save_checkpoint(
                best_pt, model, optimizer, scheduler, scaler, arc_margin,
                epoch, best_combined, combined, recall_at_1, mean_cos,
                swa_state, swa_count, no_improve_count,
            )
            logger.info(f"  ** New best: {best_combined:.6f}")
        else:
            no_improve_count += 1
            logger.info(f"  No improvement ({no_improve_count}/{patience})")

        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # SWA final
    if swa_state is not None and swa_count > 0:
        logger.info(f"Applying SWA ({swa_count} epochs)...")
        for k in swa_state:
            if swa_state[k].is_floating_point():
                swa_state[k] /= swa_count
            else:
                swa_state[k] //= swa_count
        model.load_state_dict(swa_state)

        if val_dataset is not None:
            metrics = run_retrieval_eval(
                model=model, dataset=val_dataset, device=device,
                amp=(device.type == "cuda"), max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK, seed=SEED, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = metrics["recall@1"]
            recall_at_5 = metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            # cosine estimate from a small distill batch
            model.eval()
            cos_sum = cos_n = 0
            default_t_name = teacher_names[0]
            default_t_cache = TEACHER_REGISTRY[default_t_name]["cache_dir"]
            with torch.no_grad():
                for images, labels, paths in distill_loader:
                    images = images.to(device, non_blocking=True)
                    teacher_emb = load_teacher_embeddings(
                        paths, teachers_dict[default_t_name], device,
                        default_t_cache, teacher_name=default_t_name,
                    )
                    if hasattr(model, 'proj_heads') and model.proj_heads is not None and default_t_name in model.proj_heads:
                        backbone_feat = model.forward_backbone(images)
                        student_emb = functional.normalize(model.proj_heads[default_t_name](backbone_feat), p=2, dim=1)
                    else:
                        student_emb = model.encode(images)
                    teacher_emb = teacher_emb.to(device=device, dtype=student_emb.dtype)
                    cos = functional.cosine_similarity(student_emb, teacher_emb, dim=1)
                    cos_sum += cos.sum().item()
                    cos_n += len(cos)
                    if cos_n >= 2000:
                        break
            mean_cos = cos_sum / max(cos_n, 1)
            combined = compute_combined_metric(recall_at_1, mean_cos)
            logger.info(f"SWA final: recall@1={recall_at_1:.4f} cos={mean_cos:.4f} combined={combined:.6f}")

        swa_ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "combined_metric": combined,
            "recall_at_1": recall_at_1,
            "mean_cosine": mean_cos,
            "swa_count": swa_count,
        }
        if arc_margin is not None:
            swa_ckpt["arc_margin_state_dict"] = arc_margin.state_dict()
        swa_path = out_dir / "swa_best.pt"
        torch.save(swa_ckpt, swa_path)
        logger.info(f"Saved SWA to {swa_path}")

        if combined > best_combined:
            torch.save(swa_ckpt, best_pt)
            best_combined = combined
            logger.info(f"SWA beat best: {best_combined:.6f}")

    # Final summary
    elapsed = time.time() - t_start
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    metrics = {
        "status": "success",
        "version": "v2",
        "combined_metric": combined,
        "best_combined": best_combined,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mean_cosine": mean_cos,
        "distill_loss": stats.distill_loss,
        "peak_vram_mb": peak_vram_mb,
        "epochs_trained": epoch + 1,
        "max_epochs": max_epochs,
        "early_stopped": no_improve_count >= patience,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_dir / "metrics_final_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"V2 TRAINING COMPLETE")
    print(f"  Epochs:   {epoch + 1}/{max_epochs}")
    print(f"  Best:     {best_combined:.6f} (baseline v1: 0.8588)")
    print(f"  Final:    {combined:.6f}")
    print(f"  recall@1: {recall_at_1:.6f}")
    print(f"  cosine:   {mean_cos:.6f}")
    print(f"  VRAM:     {peak_vram_mb:.1f} MB")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 LCNet student distillation")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--swa-epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        main(
            max_epochs=args.max_epochs,
            patience=args.patience,
            swa_epochs=args.swa_epochs,
            resume=args.resume,
        )
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        logger.error(f"OOM at {peak:.0f}MB VRAM")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
