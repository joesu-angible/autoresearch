"""Finetune with Qwen distillation + ArcFace on retail product checkout dataset.

Changes from arcface2:
- Removed ImageNet mixing
- Distillation: uses all product_code_dataset samples
- ArcFace: uses retail_product_checkout_crop (30 samples per class)
"""

from __future__ import annotations

import os
import site

# Ensure CUDA 12 libs from pip nvidia packages are on LD_LIBRARY_PATH (for onnxruntime-gpu)
_site_pkgs = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(os.path.dirname(os.__file__), "site-packages")
_nvidia_base = os.path.join(_site_pkgs, "nvidia")
if os.path.isdir(_nvidia_base):
    _lib_dirs = [
        os.path.join(_nvidia_base, sub, "lib")
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "cufft", "nvjitlink")
        if os.path.isdir(os.path.join(_nvidia_base, sub, "lib"))
    ]
    if _lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import argparse
import hashlib
import json
import numpy as np
import random
import sys
import time
import timm
import onnxruntime as ort
import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from transformers import AutoModel, PreTrainedModel


class PadToSquare:
    def __init__(self, color: int = 255) -> None:
        self.color = color

    def __call__(self, img: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        width, height = img.size
        if self.color != -1:
            padding = abs(width - height) // 2
            if width < height:
                return tf.pad(
                    img, (padding, 0, padding + (height - width) % 2, 0), fill=self.color, padding_mode="constant"
                )
            elif width > height:
                return tf.pad(
                    img, (0, padding, 0, padding + (width - height) % 2), fill=self.color, padding_mode="constant"
                )
        return img


class TrendyolEmbedder:
    def __init__(
        self,
        onnx_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        if onnx_path is None:
            onnx_path = "/data/mnt/mnt_ml_shared/joesu/reid/distill_qwen_lcnet050_retail_2.onnx"
            # onnx_path = "/workspace/lcnet050_pfc_supcon_f256_224_20260309_onnx_fp32.onnx"

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        elif "CPUExecutionProvider" in available_providers:
            providers = ["CPUExecutionProvider"]
        else:
            providers = available_providers[:1]

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        logger.info(f"TrendyolEmbedder: using provider {self.session.get_providers()[0]}")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        logger.info(f"TrendyolEmbedder: loaded ONNX model from {onnx_path}")

    def get_feature_dim(self) -> int:
        return 256

    def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
        if not images:
            return []

        try:
            pil_images = []
            for image in images:
                if isinstance(image, np.ndarray):
                    image_rgb = image[:, :, ::-1] if len(image.shape) == 3 and image.shape[2] == 3 else image
                    pil_image = Image.fromarray(image_rgb).convert("RGB")
                else:
                    pil_image = image.convert("RGB")
                pil_images.append(pil_image)

            input_tensors = np.stack([self.transform(img).numpy() for img in pil_images])

            input_name = self.session.get_inputs()[0].name
            embeddings = self.session.run(None, {input_name: input_tensors})[0]

            return [emb.flatten() for emb in embeddings]

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(images)

    def extract_features(self, image_crops: list[np.ndarray]) -> np.ndarray:
        if not image_crops:
            return np.array([])

        embeddings = self.encode_batch(image_crops)
        valid_embeddings = [emb for emb in embeddings if emb is not None]

        if not valid_embeddings:
            return np.array([])

        return np.array(valid_embeddings, dtype=np.float32)

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if features1.size == 0 or features2.size == 0:
            return 0.0

        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(features1, features2) / (norm1 * norm2))


def _patch_transformers_compat() -> None:
    """Monkey-patch transformers 5.x compat for custom HF models."""
    if getattr(PreTrainedModel, "_compat_patched", False):
        return
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self: PreTrainedModel, loading_info: dict) -> None:  # type: ignore[override]
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, loading_info)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched  # type: ignore[assignment]
    PreTrainedModel._compat_patched = True  # type: ignore[attr-defined]


class DINOv2Teacher:
    """Trendyol DINO v2 as teacher. Same encode_batch interface as TrendyolEmbedder."""

    def __init__(self, model_name: str = "Trendyol/trendyol-dino-v2-ecommerce-256d", device: str = "cuda") -> None:
        import os
        os.environ["XFORMERS_DISABLED"] = "1"
        _patch_transformers_compat()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"DINOv2Teacher: loaded {model_name}, output_dim=256")

    @torch.no_grad()
    def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
        if not images:
            return []
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.amp.autocast(self.device):
            out = self.model(batch)
        emb = out.last_hidden_state  # already L2-normalized 256d
        return [e.cpu().numpy() for e in emb]


class PadToSquare:
    def __init__(self, color: int = 255) -> None:
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        if self.color != -1:
            padding = abs(width - height) // 2
            if width < height:
                return TF.pad(
                    img, (padding, 0, padding + (height - width) % 2, 0), fill=self.color, padding_mode="constant"
                )
            elif width > height:
                return TF.pad(
                    img, (0, padding, 0, padding + (width - height) % 2), fill=self.color, padding_mode="constant"
                )
        return img


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = functional.linear(functional.normalize(embeddings), functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * np.cos(self.m) - sine * np.sin(self.m)
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class RandomQualityDegradation:
    """Randomly degrade image quality by downsampling and JPEG compression."""

    def __init__(
        self,
        prob: float = 0.5,
        downsample_ratio: tuple[float, float] = (0.3, 0.6),
        quality_range: tuple[int, int] = (50, 80),
    ) -> None:
        self.prob = prob
        self.downsample_ratio = downsample_ratio
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        ratio = random.uniform(*self.downsample_ratio)
        new_w, new_h = max(1, int(img.width * ratio)), max(1, int(img.height * ratio))
        orig_size = (img.width, img.height)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        img = img.resize(orig_size, Image.Resampling.BILINEAR)
        return img


class DistillImageFolder(datasets.ImageFolder):
    """ImageFolder that returns (image, label, path)."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        self.return_path = return_path

    def __getitem__(self, index: int) -> tuple[Image.Image, int] | tuple[Image.Image, int, str]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.return_path:
            return sample, target, path
        return sample, target


class SampledImageFolder(Dataset):
    """ImageFolder with max N samples per class."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        max_per_class: int = 100,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.max_per_class = max_per_class
        self.return_path = return_path

        # Find all classes
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Sample up to max_per_class from each class
        self.samples: list[tuple[str, int]] = []
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            image_files += list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPEG"))

            if len(image_files) > max_per_class:
                image_files = random.sample(image_files, max_per_class)

            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))

        logger.info(
            f"SampledImageFolder: {len(self.classes)} classes, {len(self.samples)} samples (max {max_per_class}/class)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, target, path
        return img, target


class CombinedDistillDataset(Dataset):
    """Mix product_code_dataset (train+val) with retail dataset for distillation.

    Uses random replacement: retail_ratio % of samples are replaced with retail.
    Returns (image, label, path) for teacher embedding lookup.
    """

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        transform: Callable | None = None,
        retail_ratio: float = 0.3,
        blacklist_root: str | None = None,
        blacklist_ratio: float = 0.0,
        skip_classes: set[str] | None = None,
        quality_degradation: Callable | None = None,
        skip_degradation_paths: list[str] | None = None,
    ) -> None:
        self.transform = transform
        self.retail_ratio = retail_ratio
        self.blacklist_ratio = blacklist_ratio
        self.quality_degradation = quality_degradation
        self.skip_degradation_paths = skip_degradation_paths or []
        self.samples: list[tuple[str, int]] = []
        self.retail_samples: list[tuple[str, int]] = []
        self.blacklist_samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        retail_path = Path(retail_root)

        # Collect all primary classes from all roots
        all_classes: set[str] = set()
        for root in primary_roots:
            root_path = Path(root)
            if root_path.exists():
                for d in root_path.iterdir():
                    if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                        all_classes.add(d.name)

        # Retail dataset classes (with prefix to avoid collision)
        retail_class_dirs: list[Path] = []
        if retail_path.exists():
            retail_class_dirs = sorted(
                [d for d in retail_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
            )
        retail_classes = [f"retail_{d.name}" for d in retail_class_dirs]

        # Blacklist classes (with prefix)
        bl_class_dirs: list[Path] = []
        if blacklist_root:
            bl_path = Path(blacklist_root)
            if bl_path.exists():
                bl_class_dirs = sorted(
                    [d for d in bl_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
                )
        bl_classes = [f"bl_{d.name}" for d in bl_class_dirs]

        self.classes = sorted(all_classes) + retail_classes + bl_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.blacklist_class_indices: set[int] = {self.class_to_idx[c] for c in bl_classes}

        # Load primary dataset samples from all roots (full)
        primary_count = 0
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            class_dirs = sorted(
                [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip]
            )
            for class_dir in class_dirs:
                class_idx = self.class_to_idx[class_dir.name]
                image_files = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.png"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.JPEG"))
                )
                for img_path in image_files:
                    self.samples.append((str(img_path), class_idx))
                    primary_count += 1

        # Load retail dataset samples (for random replacement)
        retail_count = 0
        for class_dir in retail_class_dirs:
            class_name = f"retail_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                self.retail_samples.append((str(img_path), class_idx))
                retail_count += 1

        # Load blacklist samples (capped to limit teacher cache misses)
        max_bl_samples = 50_000  # ~50K is enough for 10% ratio sampling
        all_bl: list[tuple[str, int]] = []
        for class_dir in bl_class_dirs:
            class_name = f"bl_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                all_bl.append((str(img_path), class_idx))
        if len(all_bl) > max_bl_samples:
            all_bl = random.sample(all_bl, max_bl_samples)
        self.blacklist_samples = all_bl
        bl_count = len(all_bl)

        logger.info(
            f"CombinedDistillDataset: {len(self.classes)} classes, "
            f"{len(self.samples)} primary, {len(self.retail_samples)} retail, "
            f"{len(self.blacklist_samples)} blacklist, "
            f"retail_ratio={retail_ratio}, blacklist_ratio={blacklist_ratio}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        # Randomly replace with blacklist or retail sample
        r = random.random()
        if self.blacklist_samples and r < self.blacklist_ratio:
            path, target = random.choice(self.blacklist_samples)
        elif self.retail_samples and r < self.blacklist_ratio + self.retail_ratio:
            path, target = random.choice(self.retail_samples)
        else:
            path, target = self.samples[index]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")

        # Apply quality degradation only if path is not in skip list
        if self.quality_degradation is not None:
            should_skip = any(skip_path in path for skip_path in self.skip_degradation_paths)
            if not should_skip:
                img = self.quality_degradation(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


class CombinedArcFaceDataset(Dataset):
    """Combine multiple primary datasets + retail dataset for ArcFace.

    Uses first primary_root's class IDs as the reference set.
    Other primary roots only include classes that exist in the reference set.
    """

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        transform: Callable | None = None,
        retail_max_per_class: int = 100,
        skip_classes: set[str] | None = None,
        quality_degradation: Callable | None = None,
        skip_degradation_paths: list[str] | None = None,
    ) -> None:
        self.transform = transform
        self.quality_degradation = quality_degradation
        self.skip_degradation_paths = skip_degradation_paths or []
        self.samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        if not primary_roots:
            raise ValueError("primary_roots must contain at least one path")

        # Collect reference class IDs from ALL primary roots (union)
        ref_class_ids = set()
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for d in root_path.iterdir():
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                    ref_class_ids.add(d.name)

        logger.info(f"Reference class IDs (union of {len(primary_roots)} roots): {len(ref_class_ids)} classes")

        # Collect all valid class directories from all primary roots
        all_classes: list[str] = []
        primary_class_dirs: list[Path] = []

        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                logger.warning(f"Primary root not found: {root}")
                continue
            for d in sorted(root_path.iterdir()):
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name in ref_class_ids:
                    primary_class_dirs.append(d)

        # Deduplicate class names while keeping all directories
        seen_class_names = set()
        for d in primary_class_dirs:
            class_name = f"primary_{d.name}"
            if class_name not in seen_class_names:
                all_classes.append(class_name)
                seen_class_names.add(class_name)

        # Retail dataset classes
        retail_path = Path(retail_root)
        retail_class_dirs = sorted(
            [d for d in retail_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
        )
        retail_classes = [f"retail_{d.name}" for d in retail_class_dirs]
        all_classes.extend(retail_classes)

        self.classes = all_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

        # Load primary dataset samples from all roots
        primary_count = 0
        for class_dir in primary_class_dirs:
            class_name = f"primary_{class_dir.name}"
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                primary_count += 1

        # Load retail dataset samples (sampled)
        retail_count = 0
        for class_dir in retail_class_dirs:
            class_name = f"retail_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            if len(image_files) > retail_max_per_class:
                image_files = random.sample(image_files, retail_max_per_class)
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                retail_count += 1

        logger.info(
            f"CombinedArcFaceDataset: {len(self.classes)} classes total "
            f"(primary={len(seen_class_names)}, retail={len(retail_classes)}), "
            f"{len(self.samples)} samples (primary={primary_count}, retail={retail_count})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")

        # Apply quality degradation only if path is not in skip list
        if self.quality_degradation is not None:
            should_skip = any(skip_path in path for skip_path in self.skip_degradation_paths)
            if not should_skip:
                img = self.quality_degradation(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


def collate_distill(
    batch: Sequence[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate for distillation dataset (img, label, path)."""
    imgs, labels, paths = [], [], []
    for img, label, path in batch:
        imgs.append(img)
        labels.append(int(label))
        paths.append(path)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), paths


def collate_arcface(
    batch: Sequence[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate for ArcFace dataset (img, label, path)."""
    imgs, labels, paths = [], [], []
    for img, label, path in batch:
        imgs.append(img)
        labels.append(int(label))
        paths.append(path)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), paths


class FrozenBackboneWithHead(nn.Module):
    """Student model with frozen backbone + trainable projection head."""

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            in_features = out.shape[-1]
        logger.info(f"Backbone {model_name} output dim: {in_features}")

        self.proj = ProjectionHead(in_features, embedding_dim)

    def train(self, mode: bool = True) -> FrozenBackboneWithHead:
        super().train(mode)
        self.backbone.eval()
        return self

    def unfreeze_last_stage(self) -> None:
        """Unfreeze the last stage of the backbone."""
        for i in [-1, -2, -3, -4]:
            if hasattr(self.backbone, "stages"):
                for p in self.backbone.stages[i].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "features"):
                for p in self.backbone.features[i].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "blocks"):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True
        self.backbone.train()

    def forward_embeddings_train(self, images: torch.Tensor) -> torch.Tensor:
        has_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if has_trainable:
            features = self.backbone(images)
        else:
            with torch.no_grad():
                features = self.backbone(images)
        emb = self.proj(features)
        emb = functional.normalize(emb, p=2, dim=1)
        return emb

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            features = self.backbone(images)
            emb = self.proj(features)
            emb = functional.normalize(emb, p=2, dim=1)
        return emb


_TEACHER_MEM_CACHE: dict[str, np.ndarray] = {}


def load_teacher_embeddings(
    image_paths: Sequence[str],
    teacher: TrendyolEmbedder,
    device: torch.device,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Load teacher embeddings with in-memory + disk caching and batch inference."""
    embeddings: list[np.ndarray] = [None] * len(image_paths)  # type: ignore[list-item]

    # Phase 1: in-memory cache → disk cache
    uncached_indices: list[int] = []
    for i, path in enumerate(image_paths):
        if path in _TEACHER_MEM_CACHE:
            embeddings[i] = _TEACHER_MEM_CACHE[path]
            continue
        if cache_dir:
            cache_path = Path(cache_dir) / f"{hashlib.md5(path.encode()).hexdigest()}.npy"
            if cache_path.exists():
                emb = np.load(cache_path)
                _TEACHER_MEM_CACHE[path] = emb
                embeddings[i] = emb
                continue
        uncached_indices.append(i)

    # Phase 2: batch inference for uncached images
    if uncached_indices:
        pil_images = [Image.open(image_paths[i]).convert("RGB") for i in uncached_indices]
        emb_list = teacher.encode_batch(pil_images)

        for j, i in enumerate(uncached_indices):
            emb = emb_list[j]
            embeddings[i] = emb
            _TEACHER_MEM_CACHE[image_paths[i]] = emb
            if cache_dir and emb is not None:
                cache_path = Path(cache_dir) / f"{hashlib.md5(image_paths[i].encode()).hexdigest()}.npy"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, emb)

    return torch.tensor(np.stack(embeddings), device=device, dtype=torch.float32)


def save_batch_visualization(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "Training Batch",
    max_images: int = 16,
    denormalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    denormalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> None:
    """Save a grid visualization of batch images.

    Args:
        images: Tensor of shape (B, C, H, W)
        labels: Tensor of shape (B,)
        output_path: Path to save the visualization
        title: Title for the plot
        max_images: Maximum number of images to show
        denormalize_mean: Mean used for normalization
        denormalize_std: Std used for normalization
    """
    import matplotlib.pyplot as plt

    n = min(len(images), max_images)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    mean = torch.tensor(denormalize_mean).view(3, 1, 1)
    std = torch.tensor(denormalize_std).view(3, 1, 1)

    for i in range(n):
        row, col = i // ncols, i % ncols
        ax = axes[row][col]

        # Denormalize
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}", fontsize=8)
        ax.axis("off")

    # Hide empty subplots
    for i in range(n, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row][col].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved batch visualization to {output_path}")


def vat_embedding_loss(
    model: FrozenBackboneWithHead,
    x: torch.Tensor,
    epsilon: float = 2.0,
    xi: float = 0.1,
    num_power_iter: int = 1,
) -> torch.Tensor:
    """Feature-level VAT loss (Miyato et al., 2018 — arXiv 1704.03976).

    Perturbs backbone features (not raw pixels) to find adversarial
    direction, then penalises cosine distance between clean and perturbed
    embeddings.  Operating in ~512-dim feature space instead of 150K-dim
    pixel space is faster and finds better adversarial directions.
    """
    # Get clean backbone features and embedding
    with torch.no_grad():
        feat_clean = model.backbone(x)
        emb_clean = functional.normalize(model.proj(feat_clean), p=2, dim=1)

    # Random unit vector in feature space
    d = torch.randn_like(feat_clean)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-12)

    for _ in range(num_power_iter):
        d = d.detach().requires_grad_(True)
        emb_perturbed = functional.normalize(model.proj(feat_clean.detach() + xi * d), p=2, dim=1)
        dist = (1.0 - functional.cosine_similarity(emb_clean, emb_perturbed, dim=1)).mean()
        (grad_d,) = torch.autograd.grad(dist, d)
        d = grad_d.detach()
        d = d / (d.norm(dim=1, keepdim=True) + 1e-12)

    # Final VAT loss with adversarial perturbation
    r_adv = epsilon * d.detach()
    emb_adv = functional.normalize(model.proj(feat_clean.detach() + r_adv), p=2, dim=1)
    return (1.0 - functional.cosine_similarity(emb_clean, emb_adv, dim=1)).mean()


@dataclass
class EpochStats:
    loss: float
    distill_loss: float
    arc_loss: float
    vat_loss: float
    sep_loss: float
    mean_cosine: float


def run_train_epoch(
    model: FrozenBackboneWithHead,
    distill_loader: DataLoader,
    arcface_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler,
    teacher: TrendyolEmbedder,
    device: torch.device,
    amp: bool,
    arc_margin: ArcMarginProduct | None,
    arc_loss_weight: float,
    cache_dir: str | None,
    drop_hard_ratio: float = 0.0,
    vat_weight: float = 0.0,
    vat_epsilon: float = 0.1,
    sep_weight: float = 0.0,
    blacklist_class_indices: set[int] | None = None,
    wl_centroid_ema: dict | None = None,
    backbone_unfrozen: bool = False,
    save_first_batch_path: Path | None = None,
) -> EpochStats:
    """Run one training epoch with separate distillation and ArcFace data."""
    model.train()

    total_loss = 0.0
    total_distill = 0.0
    total_arc = 0.0
    total_vat = 0.0
    total_sep = 0.0
    total_align = 0.0
    _bl_idx = blacklist_class_indices or set()
    n = 0
    first_batch_saved = False

    # Create iterators
    arcface_iter = iter(arcface_loader) if arcface_loader else None

    for images, labels, paths in distill_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Save first batch visualization
        if save_first_batch_path and not first_batch_saved:
            save_batch_visualization(
                images, labels, save_first_batch_path / "distill_batch.png", title="Distillation Batch (First)"
            )
            first_batch_saved = True

        # Load teacher embeddings for distillation
        teacher_emb = load_teacher_embeddings(paths, teacher, device, cache_dir)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            student_emb = model.forward_embeddings_train(images)

            # --- Distillation Loss ---
            teacher_emb = teacher_emb.to(device=device, dtype=student_emb.dtype)
            cosine = functional.cosine_similarity(student_emb, teacher_emb, dim=1)
            distill_loss = (1.0 - cosine).mean()
            batch_align = float(cosine.mean().item())

            # --- ArcFace Loss (from retail dataset) ---
            arc_loss = torch.tensor(0.0, device=device)
            if arc_margin is not None and arcface_iter is not None:
                try:
                    arc_images, arc_labels, _arc_paths = next(arcface_iter)
                except StopIteration:
                    arcface_iter = iter(arcface_loader)
                    arc_images, arc_labels, _arc_paths = next(arcface_iter)

                arc_images = arc_images.to(device, non_blocking=True)
                arc_labels = arc_labels.to(device, non_blocking=True)

                # Save first ArcFace batch visualization
                if save_first_batch_path and n == 0:
                    save_batch_visualization(
                        arc_images,
                        arc_labels,
                        save_first_batch_path / "arcface_batch.png",
                        title="ArcFace Batch (First)",
                    )

                arc_emb = model.forward_embeddings_train(arc_images)

                # ArcFace classification loss
                arc_logits = arc_margin(arc_emb, arc_labels)
                per_sample_arc = functional.cross_entropy(arc_logits, arc_labels, reduction="none")

                if drop_hard_ratio > 0.0:
                    keep = max(int(len(per_sample_arc) * (1 - drop_hard_ratio)), 1)
                    trimmed, _ = torch.topk(per_sample_arc, k=keep, largest=False)
                    arc_loss = trimmed.mean()
                else:
                    arc_loss = per_sample_arc.mean()

                # Distillation on ArcFace batch (disabled for speed — uncomment to enable)
                # arc_teacher_emb = load_teacher_embeddings(arc_paths, teacher, device, cache_dir)
                # arc_teacher_emb = arc_teacher_emb.to(device=device, dtype=arc_emb.dtype)
                # arc_distill_loss = (1.0 - functional.cosine_similarity(arc_emb, arc_teacher_emb, dim=1)).mean()

            # --- Separation Loss: push blacklist away from whitelist ---
            l_sep = torch.tensor(0.0, device=device)
            if sep_weight > 0 and _bl_idx:
                bl_mask = torch.tensor([int(lab) in _bl_idx for lab in labels], device=device)
                wl_mask = ~bl_mask

                # Update EMA whitelist centroid
                if wl_mask.any() and wl_centroid_ema is not None:
                    wl_mean = student_emb[wl_mask].detach().mean(dim=0)
                    if wl_centroid_ema.get("centroid") is None:
                        wl_centroid_ema["centroid"] = wl_mean
                    else:
                        wl_centroid_ema["centroid"] = 0.9 * wl_centroid_ema["centroid"] + 0.1 * wl_mean

                # Compute separation: blacklist vs EMA centroid (always available)
                if bl_mask.any() and wl_centroid_ema is not None and wl_centroid_ema.get("centroid") is not None:
                    bl_emb = student_emb[bl_mask]
                    centroid = functional.normalize(wl_centroid_ema["centroid"].unsqueeze(0), p=2, dim=1)
                    l_sep = (bl_emb @ centroid.T).clamp(min=0).mean()

            loss = distill_loss + arc_loss_weight * arc_loss + sep_weight * l_sep

        # --- VAT Loss (fp32, outside autocast to avoid precision issues) ---
        # Skip VAT while backbone is frozen — perturbations have no effect.
        l_vat = torch.tensor(0.0, device=device)
        if vat_weight > 0 and backbone_unfrozen:
            l_vat = vat_embedding_loss(model, images, epsilon=vat_epsilon)
            loss = loss + vat_weight * l_vat

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_distill += float(distill_loss.item())
        total_arc += float(arc_loss.item())
        total_vat += float(l_vat.item())
        total_sep += float(l_sep.item())
        total_align += batch_align
        n += 1

        # ML-style progress bar
        total_steps = len(distill_loader)
        if total_steps > 0:
            bar_len = 30
            filled = bar_len * n // total_steps
            bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
            avg_loss = total_loss / n
            avg_cos = total_align / n
            print(f"\r  {n}/{total_steps} [{bar}] - loss: {avg_loss:.4f} - cos: {avg_cos:.4f}", end="", flush=True)
    print()  # newline after epoch

    return EpochStats(
        loss=total_loss / max(n, 1),
        distill_loss=total_distill / max(n, 1),
        arc_loss=total_arc / max(n, 1),
        vat_loss=total_vat / max(n, 1),
        sep_loss=total_sep / max(n, 1),
        mean_cosine=total_align / max(n, 1),
    )


@torch.no_grad()
def run_retrieval_eval(
    model: FrozenBackboneWithHead,
    dataset: datasets.ImageFolder,
    device: torch.device,
    amp: bool,
    max_samples: int,
    topk: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Run retrieval evaluation on a validation dataset."""
    n_total = len(dataset)
    n_use = min(max_samples, n_total)
    if n_use < 2:
        raise ValueError(f"Not enough samples for retrieval eval: n={n_use}")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)[:n_use].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.eval()
    embs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            e = model.encode(images)
        embs.append(e.detach().float().cpu())
        labels.append(y.detach().cpu())

    emb = torch.cat(embs, dim=0)
    lab = torch.cat(labels, dim=0)
    emb = functional.normalize(emb, dim=1)

    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))

    k = min(int(topk), sim.size(1) - 1)
    _, nn_idx = torch.topk(sim, k=k, dim=1)
    nn_lab = lab[nn_idx]

    correct = (nn_lab == lab.view(-1, 1)).any(dim=1).float()
    recall_at_k = float(correct.mean().item())

    _, nn1_idx = torch.topk(sim, k=1, dim=1)
    nn1_lab = lab[nn1_idx.squeeze(1)]
    recall_at_1 = float((nn1_lab == lab).float().mean().item())

    return {
        "retrieval_n": float(n_use),
        "recall@1": recall_at_1,
        f"recall@{k}": recall_at_k,
    }


def build_transform(
    model_name: str,
    image_size: int,
    is_training: bool = False,
    quality_degradation_prob: float = 0.5,
) -> transforms.Compose:
    """Build transform for student model."""
    tmp_model = timm.create_model(model_name, pretrained=True)
    data_config = resolve_data_config(tmp_model.pretrained_cfg)
    mean = data_config["mean"]
    std = data_config["std"]

    if is_training:
        return transforms.Compose(
            [
                # RandomQualityDegradation moved to Dataset level for path-based control
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)], p=0.4),
                PadToSquare(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-dir",
        type=str,
        # default="/data/mnt/mnt_ml_shared/visualization/1229/original_crop",
        default="/data/mnt/mnt_ml_shared/Vic/product_code_dataset/train",
        help="Distillation dataset directory.",
    )
    p.add_argument(
        "--val-dir",
        type=str,
        default="/data/mnt/mnt_ml_shared/Vic/product_code_dataset/val",
        help="Validation dataset directory.",
    )
    p.add_argument(
        "--arcface-dir",
        type=str,
        default="/data/mnt/mnt_ml_shared/Vic/retail_product_checkout_crop",
        help="ArcFace dataset directory (retail product checkout crops).",
    )
    p.add_argument(
        "--arcface-max-per-class",
        type=int,
        default=100,
        help="Max samples per class for ArcFace dataset.",
    )
    p.add_argument("--model-name", type=str, default="hf-hub:timm/lcnet_050.ra2_in1k")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--arcface-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val-split", type=float, default=0.0)
    p.add_argument("--quality-degradation-prob", type=float, default=0.5)
    p.add_argument(
        "--drop-hard-ratio",
        type=float,
        default=0.2,
        help="Fraction of hardest samples to drop in ArcFace loss.",
    )
    p.add_argument("--use-arcface", action="store_true", default=True)
    p.add_argument("--no-arcface", action="store_false", dest="use_arcface")
    p.add_argument("--arcface-s", type=float, default=32.0)
    p.add_argument("--arcface-m", type=float, default=0.50)
    p.add_argument("--arcface-loss-weight", type=float, default=0.05)
    p.add_argument("--vat-weight", type=float, default=0, help="VAT loss weight (0 to disable)")
    p.add_argument("--vat-epsilon", type=float, default=8.0, help="VAT perturbation magnitude")
    p.add_argument("--sep-weight", type=float, default=1, help="Blacklist-whitelist separation loss weight")
    p.add_argument("--arcface-phaseout-epoch", type=int, default=0,
                   help="Epoch at which ArcFace weight starts linearly decaying to 0. 0=disabled (constant weight).")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 to disable)")
    p.add_argument(
        "--teacher-model-name",
        type=str,
        default="Trendyol/trendyol-dino-v2-ecommerce-256d",
    )
    p.add_argument("--use-dino-teacher", action="store_true", default=False, help="Use Trendyol DINO v2 as teacher instead of ONNX student model")
    p.add_argument("--unfreeze-epoch", type=int, default=5)
    p.add_argument(
        "--teacher-cache-dir",
        type=str,
        default="workspace/output/trendyol_teacher_cache2",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="workspace/output/distill_trendyol_lcnet050_retail",
    )
    p.add_argument("--retrieval-max-samples", type=int, default=5000)
    p.add_argument("--retrieval-topk", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    arcface_dir = Path(args.arcface_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build transforms
    train_transform = build_transform(
        args.model_name, args.image_size, is_training=True, quality_degradation_prob=args.quality_degradation_prob
    )

    # ReID dataset root
    reid_root = Path("/data/mnt/mnt_ml_shared/joesu/reid/data/reid_train/train")
    reid_products = str(reid_root / "products")
    reid_commodity = str(reid_root / "commodity")
    reid_negatives = str(reid_root / "negatives")

    # Distillation dataset: products + commodity (positive) + negatives (hard negative)
    val_dir = Path(args.val_dir)
    quality_degradation = RandomQualityDegradation(prob=args.quality_degradation_prob)
    distill_dataset = CombinedDistillDataset(
        primary_roots=[
            reid_products,
            reid_commodity,
        ],
        retail_root=str(arcface_dir),
        blacklist_root=reid_negatives,
        blacklist_ratio=0.10,
        skip_classes={"0000000000"},
        transform=train_transform,
        quality_degradation=quality_degradation,
        skip_degradation_paths=[],
    )

    # ArcFace dataset: products only (need barcode for classification)
    arcface_dataset: CombinedArcFaceDataset | None = None
    if args.use_arcface:
        if not arcface_dir.exists():
            logger.warning(f"ArcFace retail dir not found: {arcface_dir}, using only primary dataset")
        # Exclude val barcodes from ArcFace to prevent leakage
        val_barcodes = {
            d.name for d in val_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        } if val_dir.exists() else set()
        arcface_skip = {"0000000000"} | val_barcodes
        logger.info(f"ArcFace: skipping {len(arcface_skip)} classes (1 empty + {len(val_barcodes)} val barcodes)")

        arcface_dataset = CombinedArcFaceDataset(
            primary_roots=[
                reid_products,
            ],
            retail_root=str(arcface_dir),
            transform=train_transform,
            retail_max_per_class=args.arcface_max_per_class,
            skip_classes=arcface_skip,
            quality_degradation=quality_degradation,
            skip_degradation_paths=[],
        )

    # DataLoaders
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_distill,
    )

    arcface_loader: DataLoader | None = None
    if arcface_dataset is not None:
        arcface_loader = DataLoader(
            arcface_dataset,
            batch_size=args.arcface_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            collate_fn=collate_arcface,
        )

    # Model
    model = FrozenBackboneWithHead(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        device=str(device),
    ).to(device)

    # ArcFace head (uses arcface dataset classes)
    arc_margin: ArcMarginProduct | None = None
    if args.use_arcface and arcface_dataset is not None:
        arc_margin = ArcMarginProduct(
            in_features=args.embedding_dim,
            out_features=len(arcface_dataset.classes),
            s=args.arcface_s,
            m=args.arcface_m,
        ).to(device)
        logger.info(f"ArcFace enabled: {len(arcface_dataset.classes)} classes, s={args.arcface_s}, m={args.arcface_m}")

    # Teacher model
    if args.use_dino_teacher:
        logger.info(f"Loading DINO v2 teacher: {args.teacher_model_name}")
        teacher = DINOv2Teacher(model_name=args.teacher_model_name, device=str(device))
    else:
        logger.info("Loading ONNX teacher (TrendyolEmbedder)")
        teacher = TrendyolEmbedder(device=str(device))

    # Unfreeze backbone from the start — use differential lr
    model.unfreeze_last_stage()
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.proj.parameters())
    if arc_margin is not None:
        head_params += list(arc_margin.parameters())

    optimizer = torch.optim.SGD(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * 0.1},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Validation dataset for retrieval eval
    val_transform = build_transform(args.model_name, args.image_size, is_training=False)
    val_dataset: datasets.ImageFolder | None = None
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
        logger.info(f"Validation dataset: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")

    # Training loop
    best_recall = 0.0
    best_cosine = 0.0
    no_improve = 0
    main._best_combined = 0.0
    wl_centroid_ema: dict = {"centroid": None}
    for epoch in range(args.epochs):
        t0 = time.time()
        # Save first batch visualization only on epoch 0
        first_batch_path = out_dir if epoch == 0 else None

        # ArcFace phase-out: linearly decay weight to 0 after phaseout epoch
        if args.arcface_phaseout_epoch > 0 and epoch >= args.arcface_phaseout_epoch:
            remaining = args.epochs - args.arcface_phaseout_epoch
            progress = (epoch - args.arcface_phaseout_epoch) / max(remaining, 1)
            effective_arc_weight = args.arcface_loss_weight * (1.0 - progress)
        else:
            effective_arc_weight = args.arcface_loss_weight

        stats = run_train_epoch(
            model=model,
            distill_loader=distill_loader,
            arcface_loader=arcface_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            teacher=teacher,
            device=device,
            amp=(device.type == "cuda"),
            arc_margin=arc_margin,
            arc_loss_weight=effective_arc_weight,
            cache_dir=args.teacher_cache_dir,
            drop_hard_ratio=args.drop_hard_ratio,
            vat_weight=args.vat_weight,
            vat_epsilon=args.vat_epsilon,
            sep_weight=args.sep_weight,
            blacklist_class_indices=distill_dataset.blacklist_class_indices,
            wl_centroid_ema=wl_centroid_ema,
            backbone_unfrozen=True,
            save_first_batch_path=first_batch_path,
        )
        elapsed = time.time() - t0

        arc_w_str = f" arc_w={effective_arc_weight:.4f}" if effective_arc_weight != args.arcface_loss_weight else ""
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={stats.loss:.4f} distill={stats.distill_loss:.4f} "
            f"arc={stats.arc_loss:.4f} vat={stats.vat_loss:.4f} sep={stats.sep_loss:.4f} cosine={stats.mean_cosine:.4f}{arc_w_str} | "
            f"{elapsed:.1f}s"
        )

        # Retrieval evaluation
        recall_at_1 = 0.0
        if val_dataset is not None:
            retrieval_metrics = run_retrieval_eval(
                model=model,
                dataset=val_dataset,
                device=device,
                amp=(device.type == "cuda"),
                max_samples=args.retrieval_max_samples,
                topk=args.retrieval_topk,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            recall_at_1 = retrieval_metrics["recall@1"]
            logger.info(
                f"  Retrieval: recall@1={recall_at_1:.4f} "
                f"recall@{args.retrieval_topk}={retrieval_metrics.get(f'recall@{args.retrieval_topk}', 0):.4f}"
            )
            if recall_at_1 > best_recall:
                best_recall = recall_at_1

        # Early stopping: combined metric (recall@1 + cosine alignment)
        combined_metric = recall_at_1 * 0.5 + stats.mean_cosine * 0.5
        if combined_metric > getattr(main, "_best_combined", 0.0):
            main._best_combined = combined_metric
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {args.patience} epochs, "
                f"best recall@1={best_recall:.4f}, best cosine={best_cosine:.4f})"
            )
            break

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_name": args.model_name,
            "embedding_dim": args.embedding_dim,
            "backbone_state_dict": model.backbone.state_dict(),
            "proj_state_dict": model.proj.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": {
                "loss": stats.loss,
                "distill_loss": stats.distill_loss,
                "arc_loss": stats.arc_loss,
                "vat_loss": stats.vat_loss,
                "sep_loss": stats.sep_loss,
                "mean_cosine": stats.mean_cosine,
            },
        }
        if arc_margin is not None:
            ckpt["arc_margin_state_dict"] = arc_margin.state_dict()

        torch.save(ckpt, out_dir / "checkpoint_last.pt")

        if stats.mean_cosine > best_cosine:
            best_cosine = stats.mean_cosine
            torch.save(ckpt, out_dir / "checkpoint_best.pt")
            logger.info(f"  -> New best cosine: {best_cosine:.4f}")

    logger.info(f"Training complete. Best cosine: {best_cosine:.4f}, Best recall@1: {best_recall:.4f}")

    # --- Export best checkpoint to ONNX ---
    best_ckpt_path = out_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        logger.info("Exporting best checkpoint to ONNX ...")
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
        model.cpu()
        model.backbone.load_state_dict(best_ckpt["backbone_state_dict"])
        model.proj.load_state_dict(best_ckpt["proj_state_dict"])
        model.eval()

        # Wrapper that runs backbone → proj → L2 norm (matches encode())
        class _ExportWrapper(nn.Module):
            def __init__(self, backbone: nn.Module, proj: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                self.proj = proj

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                emb = self.proj(features)
                return functional.normalize(emb, p=2, dim=1)

        wrapper = _ExportWrapper(model.backbone, model.proj)
        wrapper.eval()

        dummy = torch.randn(1, 3, args.image_size, args.image_size)
        onnx_path = out_dir / "model.onnx"
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        logger.info(f"ONNX model saved to {onnx_path}")

        # Verify ONNX output matches PyTorch
        import onnxruntime as ort

        ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_np = dummy.numpy()
        ort_out = ort_sess.run(None, {ort_sess.get_inputs()[0].name: dummy_np})[0]
        pt_out = wrapper(dummy).detach().numpy()
        max_diff = np.abs(ort_out - pt_out).max()
        logger.info(f"ONNX verification: max_diff={max_diff:.6f} (should be < 1e-5)")

    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
