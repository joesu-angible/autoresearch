"""Generate the deterministic productness validation holdout list.

Selects ~10% of (REID_PRODUCTS ∪ REID_NEGATIVES) image paths via SHA-1 bucketing
on the path *relative to its dataset root* + an `is_negative` tag. Pure function
of (sorted file list, seed) — rerunning produces identical output as long as the
dataset on disk is unchanged. Relative-path hashing is portable across machines.

Output: student_finetune/productness_val_paths.txt (one absolute path per line, sorted).
The file is gitignored — regenerate by running this script. Generation is fast
(~seconds even for hundreds of thousands of files).

Why bucket by hash and not random.sample? `random.sample` depends on Python version
and platform PRNG state. Hash-bucketing is stable across machines and Python versions.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "student_finetune"))

from prepare import REID_PRODUCTS, REID_NEGATIVES  # noqa: E402

HOLDOUT_FRAC = 0.10
SEED_TAG = "productness-val-v1"
OUTPUT_PATH = REPO_ROOT / "student_finetune" / "productness_val_paths.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    paths = [str(p) for p in root_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


def stable_key(absolute_path: str, root: str, is_negative: bool) -> str:
    """Portable key: relative path + tag, so the bucket is machine-independent."""
    rel = str(Path(absolute_path).relative_to(root))
    tag = "neg" if is_negative else "pos"
    return f"{tag}|{rel}"


def in_holdout(key: str, frac: float = HOLDOUT_FRAC, seed: str = SEED_TAG) -> bool:
    """SHA-1 bucket on (seed | key). Stable across runs/machines/Python versions."""
    digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # uniform in [0,1)
    return bucket < frac


def build() -> list[str]:
    products = list_images(REID_PRODUCTS)
    negatives = list_images(REID_NEGATIVES)
    holdout = []
    for p in products:
        if in_holdout(stable_key(p, REID_PRODUCTS, is_negative=False)):
            holdout.append(p)
    for p in negatives:
        if in_holdout(stable_key(p, REID_NEGATIVES, is_negative=True)):
            holdout.append(p)
    holdout.sort()
    return holdout


def main() -> None:
    holdout = build()
    OUTPUT_PATH.write_text("\n".join(holdout) + ("\n" if holdout else ""))
    total_products = len(list_images(REID_PRODUCTS))
    total_negatives = len(list_images(REID_NEGATIVES))
    print(
        f"Wrote {len(holdout)} holdout paths to {OUTPUT_PATH} "
        f"(of {total_products + total_negatives} total: "
        f"{total_products} products + {total_negatives} negatives)"
    )


if __name__ == "__main__":
    main()
