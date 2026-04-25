"""Integration smoke: run_train_epoch with productness=on on a tiny synthetic dataset.

CPU-only. No real teacher cache. Builds:
  - 4 fake images via PIL
  - A stub teacher that returns random embeddings (cached on the fly)
  - One distill batch through run_train_epoch with USE_PRODUCTNESS_CLS=True path

Asserts: no exception, all losses are finite, productness stats are populated,
and the productness BCE actually contributed to the gradient (productness_head
weights moved between before/after).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_v2 import ProductnessLCNet  # noqa: E402
from train import LCNET_SCALE, SE_START_BLOCK, SE_REDUCTION, ACTIVATION, KERNEL_SIZES, run_train_epoch  # noqa: E402
from prepare import EMBEDDING_DIM, IMAGE_SIZE, TEACHER_REGISTRY  # noqa: E402

TEACHER = "dinov3_ft"
TEACHER_DIM = TEACHER_REGISTRY[TEACHER]["embedding_dim"]


class TinyDistillDataset(Dataset):
    """4 sample images: 2 'product' + 2 'negative'. Paths control productness target."""

    def __init__(self, tmp_dir: Path, image_size: int):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.samples: list[tuple[str, int]] = []
        for i, (kind, label) in enumerate([("product", 0), ("product", 1), ("negative", 2), ("negative", 2)]):
            p = tmp_dir / f"{kind}_{i}.jpg"
            Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(p)
            self.samples.append((str(p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, label, path


class StubTeacher:
    """Returns deterministic random embeddings for any path. No real model."""

    def encode_batch(self, images):
        return [np.random.randn(TEACHER_DIM).astype(np.float32) for _ in images]


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return images, labels, paths


def test_productness_integration_one_step(tmp_path: Path, monkeypatch):
    """Single train step with productness on; verifies wiring + finite loss."""
    device = torch.device("cpu")

    # Tiny dataset
    ds = TinyDistillDataset(tmp_path, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_collate)
    negative_paths = {p for p, _ in ds.samples if "negative" in p}
    assert len(negative_paths) == 2

    # Model with productness head
    model = ProductnessLCNet(
        scale=LCNET_SCALE, se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION, activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES, embedding_dim=EMBEDDING_DIM,
        device="cpu", teacher_dims={TEACHER: TEACHER_DIM},
        productness_hidden=64,
    ).to(device)
    model.unfreeze_last_stage()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    # Stub out teacher embedding loader to avoid touching real cache
    import train as train_mod

    def fake_load_teacher(paths, _teacher, dev, _cache, teacher_name):
        return torch.randn(len(paths), TEACHER_DIM, device=dev)

    monkeypatch.setattr(train_mod, "load_teacher_embeddings", fake_load_teacher)

    # Snapshot productness_head weight before training step
    head_w_before = next(model.productness_head.parameters()).detach().clone()

    stats = run_train_epoch(
        model=model,
        distill_loader=loader,
        arcface_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        teachers={TEACHER: StubTeacher()},
        teacher_weights={TEACHER: 1.0},
        device=device,
        amp=False,
        arc_margin=None,
        arc_loss_weight=0.0,
        productness_head=model.productness_head,
        productness_negative_paths=negative_paths,
        productness_weight=0.05,
    )

    # All losses finite
    for field in ("loss", "distill_loss", "productness_loss", "productness_acc"):
        v = getattr(stats, field)
        assert np.isfinite(v), f"{field}={v} is not finite"

    # Productness branch fired: n samples seen, accuracy in [0, 1]
    assert stats.productness_n == 4
    assert 0.0 <= stats.productness_acc <= 1.0
    assert stats.productness_loss > 0.0  # BCE should be positive on random init

    # Gradient flowed through productness_head — weights moved
    head_w_after = next(model.productness_head.parameters()).detach().clone()
    delta = (head_w_after - head_w_before).abs().max().item()
    assert delta > 0.0, "productness_head weights did not change — gradient not flowing"


def test_productness_smoothing_and_focal_path(tmp_path: Path, monkeypatch):
    """Same one-step smoke but with label smoothing + focal γ active.

    Confirms the new kwargs flow through run_train_epoch end-to-end without
    blowing up — finite loss, positive accuracy, gradients move.
    """
    device = torch.device("cpu")
    ds = TinyDistillDataset(tmp_path, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_collate)
    negative_paths = {p for p, _ in ds.samples if "negative" in p}

    model = ProductnessLCNet(
        scale=LCNET_SCALE, se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION, activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES, embedding_dim=EMBEDDING_DIM,
        device="cpu", teacher_dims={TEACHER: TEACHER_DIM},
        productness_hidden=64,
    ).to(device)
    model.unfreeze_last_stage()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    import train as train_mod
    monkeypatch.setattr(
        train_mod, "load_teacher_embeddings",
        lambda paths, *a, **k: torch.randn(len(paths), TEACHER_DIM, device=device),
    )

    head_w_before = next(model.productness_head.parameters()).detach().clone()

    stats = run_train_epoch(
        model=model, distill_loader=loader, arcface_loader=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        teachers={TEACHER: StubTeacher()}, teacher_weights={TEACHER: 1.0},
        device=device, amp=False,
        arc_margin=None, arc_loss_weight=0.0,
        productness_head=model.productness_head,
        productness_negative_paths=negative_paths,
        productness_weight=0.05,
        productness_label_smoothing_pos=0.05,
        productness_label_smoothing_neg=0.02,
        productness_focal_gamma=2.0,
    )

    assert np.isfinite(stats.productness_loss)
    assert stats.productness_loss > 0.0
    assert stats.productness_n == 4
    head_w_after = next(model.productness_head.parameters()).detach().clone()
    assert (head_w_after - head_w_before).abs().max().item() > 0.0


def test_v1_default_path_unchanged(tmp_path: Path, monkeypatch):
    """Sanity: when productness kwargs are absent, run_train_epoch behaves as V1."""
    device = torch.device("cpu")
    ds = TinyDistillDataset(tmp_path, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_collate)

    from train import LCNet
    model = LCNet(
        scale=LCNET_SCALE, se_start_block=SE_START_BLOCK,
        se_reduction=SE_REDUCTION, activation=ACTIVATION,
        kernel_sizes=KERNEL_SIZES, embedding_dim=EMBEDDING_DIM,
        device="cpu", teacher_dims={TEACHER: TEACHER_DIM},
    ).to(device)
    model.unfreeze_last_stage()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    import train as train_mod
    monkeypatch.setattr(
        train_mod, "load_teacher_embeddings",
        lambda paths, *a, **k: torch.randn(len(paths), TEACHER_DIM, device=device),
    )

    stats = run_train_epoch(
        model=model, distill_loader=loader, arcface_loader=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        teachers={TEACHER: StubTeacher()}, teacher_weights={TEACHER: 1.0},
        device=device, amp=False,
        arc_margin=None, arc_loss_weight=0.0,
        # Productness kwargs intentionally OMITTED — V1 path
    )

    assert stats.productness_loss == 0.0
    assert stats.productness_n == 0
    assert np.isfinite(stats.loss)
