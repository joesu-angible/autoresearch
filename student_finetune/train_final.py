"""Final production training script for LCNet student distillation.

Uses the optimal configuration found through 69 experiments.
Features:
  - Configurable MAX_EPOCHS (not limited to prepare.py's EPOCHS=10)
  - Early stopping with patience on combined_metric
  - Checkpoint saving every epoch: last.pt + best.pt
  - Resume from last.pt on crash/restart
  - SWA (Stochastic Weight Averaging) over configurable window

Usage:
  python train_final.py                          # Train from scratch
  python train_final.py --resume                 # Resume from last.pt
  python train_final.py --max-epochs 30          # Custom epoch count
  python train_final.py --patience 5             # Custom early stop patience
"""

import argparse
import json
import sys
import time
import numpy as np
import random
import torch
import torch.nn.functional as functional
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from prepare import (
    CombinedDistillDataset, CombinedArcFaceDataset,
    collate_distill, collate_arcface,
    PadToSquare,
    load_teacher_embeddings,
    TEACHER_REGISTRY, init_teachers, build_all_teacher_caches,
    run_retrieval_eval, compute_combined_metric,
    build_val_transform,
    build_distill_dataset, build_arcface_dataset, build_val_dataset,
    EMBEDDING_DIM, IMAGE_SIZE,
    set_seed,
)

# Import model + training components from train.py
from train import (
    LCNet, load_pretrained_lcnet, ArcMarginProduct,
    RandomQualityDegradation, build_train_transform,
    run_train_epoch, EpochStats, make_divisible,
)


# ============================================================
# OPTIMAL CONFIGURATION (from 69 experiments)
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

# ArcFace
USE_ARCFACE = True
ARCFACE_S = 32.0
ARCFACE_M = 0.30
ARCFACE_LOSS_WEIGHT = 0.03
ARCFACE_PHASEOUT_EPOCH = 3
ARCFACE_MAX_PER_CLASS = 200

# Architecture
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

# Output
OUTPUT_DIR = "/data/training/reid/workspace/output/distill_final_lcnet050"


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    path: Path,
    model: LCNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    arc_margin: ArcMarginProduct | None,
    epoch: int,
    best_combined: float,
    combined: float,
    recall_at_1: float,
    mean_cosine: float,
    swa_state: dict | None,
    swa_count: int,
    no_improve_count: int,
) -> None:
    """Save full training state for resume."""
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


def load_checkpoint(
    path: Path,
    model: LCNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    arc_margin: ArcMarginProduct | None,
    device: torch.device,
) -> dict:
    """Load full training state for resume. Returns metadata dict."""
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

    logger.info(f"=== Final Training: max_epochs={max_epochs}, patience={patience}, swa_epochs={swa_epochs} ===")

    # --- Data ---
    train_transform = build_train_transform(IMAGE_SIZE)
    quality_degradation = RandomQualityDegradation(prob=QUALITY_DEGRADATION_PROB)
    distill_dataset = build_distill_dataset(train_transform, quality_degradation)

    arcface_dataset, num_arcface_classes = None, 0
    if USE_ARCFACE:
        arcface_dataset, num_arcface_classes = build_arcface_dataset(
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
    if hasattr(distill_dataset, 'blacklist_samples') and distill_dataset.blacklist_samples:
        all_image_paths += [s[0] for s in distill_dataset.blacklist_samples]
    all_image_paths = list(set(all_image_paths))

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

    # Unfreeze backbone from start
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
        logger.info(f"  Resumed at epoch {start_epoch}, best_combined={best_combined:.6f}, "
                     f"no_improve={no_improve_count}/{patience}")

    # --- Training state ---
    wl_centroid_ema: dict = {"centroid": None}
    t_start = time.time()

    logger.info(f"Training epochs {start_epoch} -> {max_epochs}, early stop patience={patience}")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        # ArcFace phase-out: linearly decay to 0 after phaseout epoch
        if ARCFACE_PHASEOUT_EPOCH > 0 and epoch >= ARCFACE_PHASEOUT_EPOCH:
            remaining = max_epochs - ARCFACE_PHASEOUT_EPOCH
            progress = (epoch - ARCFACE_PHASEOUT_EPOCH) / max(remaining, 1)
            effective_arc_weight = ARCFACE_LOSS_WEIGHT * (1.0 - progress)
        else:
            effective_arc_weight = ARCFACE_LOSS_WEIGHT

        # --- Train one epoch ---
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
        elapsed_epoch = time.time() - t0

        arc_str = f" arc_w={effective_arc_weight:.4f}" if effective_arc_weight != ARCFACE_LOSS_WEIGHT else ""
        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"loss={stats.loss:.4f} distill={stats.distill_loss:.4f} "
            f"arc={stats.arc_loss:.4f} cosine={stats.mean_cosine:.4f}{arc_str} | "
            f"{elapsed_epoch:.1f}s"
        )

        # --- SWA accumulation ---
        if epoch >= swa_start:
            sd = {k: v.clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = sd
            else:
                for k in swa_state:
                    swa_state[k] += sd[k]
            swa_count += 1
            logger.info(f"  SWA: accumulated ({swa_count}/{swa_epochs})")

        # --- Evaluation ---
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        mean_cos = stats.mean_cosine

        if val_dataset is not None:
            retrieval_metrics = run_retrieval_eval(
                model=model, dataset=val_dataset, device=device,
                amp=(device.type == "cuda"), max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK, seed=SEED, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = retrieval_metrics["recall@1"]
            recall_at_5 = retrieval_metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            logger.info(f"  Retrieval: recall@1={recall_at_1:.4f} recall@5={recall_at_5:.4f}")

        combined = compute_combined_metric(recall_at_1, mean_cos)
        logger.info(f"  Combined: {combined:.6f} (best={best_combined:.6f})")

        # --- Checkpoint: last.pt (every epoch) ---
        save_checkpoint(
            last_pt, model, optimizer, scheduler, scaler, arc_margin,
            epoch, best_combined, combined, recall_at_1, mean_cos,
            swa_state, swa_count, no_improve_count,
        )
        logger.info(f"  Saved {last_pt}")

        # --- Checkpoint: best.pt ---
        if combined > best_combined:
            best_combined = combined
            no_improve_count = 0
            save_checkpoint(
                best_pt, model, optimizer, scheduler, scaler, arc_margin,
                epoch, best_combined, combined, recall_at_1, mean_cos,
                swa_state, swa_count, no_improve_count,
            )
            logger.info(f"  ** New best: {best_combined:.6f} -> saved {best_pt}")
        else:
            no_improve_count += 1
            logger.info(f"  No improvement ({no_improve_count}/{patience})")

        # --- Early stopping ---
        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs")
            break

    # --- Apply SWA and final eval ---
    if swa_state is not None and swa_count > 0:
        logger.info(f"Applying SWA weights (averaged over {swa_count} epochs)...")
        for k in swa_state:
            if swa_state[k].is_floating_point():
                swa_state[k] /= swa_count
            else:
                swa_state[k] //= swa_count
        model.load_state_dict(swa_state)

        # Re-evaluate SWA model
        if val_dataset is not None:
            default_t_name = teacher_names[0]
            default_t_cache = TEACHER_REGISTRY[default_t_name]["cache_dir"]
            model.eval()
            cos_sum, cos_n = 0.0, 0
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
            mean_cos = cos_sum / max(cos_n, 1)

            retrieval_metrics = run_retrieval_eval(
                model=model, dataset=val_dataset, device=device,
                amp=(device.type == "cuda"), max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK, seed=SEED, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = retrieval_metrics["recall@1"]
            recall_at_5 = retrieval_metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            combined = compute_combined_metric(recall_at_1, mean_cos)
            logger.info(f"SWA final: recall@1={recall_at_1:.4f} mean_cos={mean_cos:.4f} combined={combined:.6f}")

        # Save SWA model as swa_best.pt
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
        logger.info(f"Saved SWA model to {swa_path}")

        # If SWA is better, also overwrite best.pt
        if combined > best_combined:
            torch.save(swa_ckpt, best_pt)
            best_combined = combined
            logger.info(f"SWA beat best! Updated {best_pt} -> {best_combined:.6f}")

    # --- Final summary ---
    elapsed = time.time() - t_start
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    metrics = {
        "status": "success",
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
    with open(out_dir / "metrics_final.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Epochs: {epoch + 1}/{max_epochs}" +
          (f" (early stopped)" if no_improve_count >= patience else ""))
    print(f"  Best combined:  {best_combined:.6f}")
    print(f"  Final combined: {combined:.6f}")
    print(f"  recall@1:       {recall_at_1:.6f}")
    print(f"  mean_cosine:    {mean_cos:.6f}")
    print(f"  Peak VRAM:      {peak_vram_mb:.1f} MB")
    print(f"  Elapsed:        {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Checkpoints:    {out_dir}/{{last,best,swa_best}}.pt")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final LCNet student distillation training")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Maximum training epochs (default: 30)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience in epochs (default: 7)")
    parser.add_argument("--swa-epochs", type=int, default=3,
                        help="SWA averaging window (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt checkpoint")
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
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
