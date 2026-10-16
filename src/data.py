"""
data.py — CIFAR-100 loading and subset construction for kd-lowdata.

Design decisions:
- A fixed 5,000-image stratified validation split is carved out from the
  50k training images once, using VAL_SEED=42. This split is the same for
  ALL experiments and is NEVER used during training.
- The remaining 45k images form the "training pool". Subsets are drawn from
  this pool using get_subset_indices(fraction, seed).
- Subsets are NESTED per seed: each class's 450 training-pool indices are
  shuffled ONCE with `seed`, and the first k=ceil(450*fraction) are returned.
  This guarantees 5% ⊂ 10% ⊂ 25% ⊂ 100%.
- The same (fraction, seed) always returns identical indices, so Regime B
  (teacher and student share the same data) is trivially implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
VAL_SEED: int = 42          # NEVER change; keeps val split stable across runs
VAL_SIZE: int = 5_000       # images held out for validation
NUM_CLASSES: int = 100
TRAIN_POOL_SIZE: int = 50_000 - VAL_SIZE   # 45_000

# CIFAR-100 normalisation constants (mean / std per channel)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ── Transforms ──────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Standard CIFAR-100 training augmentation for all methods."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """Deterministic eval transform (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


# ── Val / Train-pool split ───────────────────────────────────────────────────

def _get_val_and_pool_indices(
    full_train_targets: List[int],
    val_size: int = VAL_SIZE,
    val_seed: int = VAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (val_indices, pool_indices) from the 50k training set.

    Stratified: val_size // NUM_CLASSES images per class are held out.
    Shuffling uses `val_seed` so the split is deterministic.

    Parameters
    ----------
    full_train_targets:
        Integer class labels for all 50k training images (in dataset order).
    val_size:
        Total number of images to hold out for validation.
    val_seed:
        RNG seed for the stratified sampling.

    Returns
    -------
    val_indices:
        Indices (into the 50k dataset) reserved for validation.
    pool_indices:
        Remaining indices available for training subsets.
    """
    rng = np.random.RandomState(val_seed)
    targets = np.array(full_train_targets)
    val_per_class = val_size // NUM_CLASSES  # 50 per class for val_size=5000

    val_idx: list[int] = []
    pool_idx: list[int] = []

    for cls in range(NUM_CLASSES):
        cls_indices = np.where(targets == cls)[0]
        shuffled = rng.permutation(cls_indices)
        val_idx.extend(shuffled[:val_per_class].tolist())
        pool_idx.extend(shuffled[val_per_class:].tolist())

    return np.array(val_idx, dtype=np.int64), np.array(pool_idx, dtype=np.int64)


# ── Subset selection ─────────────────────────────────────────────────────────

def get_subset_indices(
    pool_indices: np.ndarray,
    pool_targets: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Return class-balanced, nested subset indices from the training pool.

    "Nested" means the subset at fraction f is always a prefix of the subset
    at any fraction f' > f (when using the same seed). This is achieved by
    shuffling each class's pool indices once with `seed` and taking the
    first ceil(n_cls * fraction) entries.

    Parameters
    ----------
    pool_indices:
        Absolute dataset indices of the 45k training pool images.
    pool_targets:
        Class labels corresponding to `pool_indices` (same length).
    fraction:
        Fraction of the training pool to use (0 < fraction ≤ 1.0).
    seed:
        RNG seed for the per-class shuffle. Same seed ⇒ same order ⇒ nested.

    Returns
    -------
    selected:
        Sorted array of absolute dataset indices forming the subset.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")

    rng = np.random.RandomState(seed)
    targets = np.array(pool_targets)
    selected: list[int] = []

    for cls in range(NUM_CLASSES):
        cls_mask = targets == cls
        cls_pool_indices = pool_indices[cls_mask]   # absolute dataset indices
        n_cls = len(cls_pool_indices)               # ≈ 450

        # Shuffle this class's pool indices with the given seed
        perm = rng.permutation(n_cls)
        shuffled = cls_pool_indices[perm]

        # Take the first k (nested prefix)
        k = max(1, int(np.ceil(n_cls * fraction)))
        selected.extend(shuffled[:k].tolist())

    return np.sort(np.array(selected, dtype=np.int64))


# ── Main factory ─────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str | Path,
    subset_fraction: float,
    subset_seed: int,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_size: int = VAL_SIZE,
    val_seed: int = VAL_SEED,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build train / val / test DataLoaders for CIFAR-100.

    Returns
    -------
    train_loader, val_loader, test_loader, subset_indices

    `subset_indices` are the absolute indices into the 50k training dataset
    used for this run (useful for Regime B verification).
    """
    data_dir = Path(data_dir)

    # Full 50k training dataset (download if needed)
    full_train = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=get_train_transform(),
        download=download,
    )
    full_train_eval = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=get_eval_transform(),
        download=False,  # already downloaded above
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=get_eval_transform(),
        download=False,
    )

    # Split into validation pool and training pool
    val_indices, pool_indices = _get_val_and_pool_indices(
        full_train.targets, val_size=val_size, val_seed=val_seed
    )
    pool_targets = np.array(full_train.targets)[pool_indices]

    # Select subset from training pool
    subset_indices = get_subset_indices(
        pool_indices, pool_targets, fraction=subset_fraction, seed=subset_seed
    )

    logger.info(
        "DataLoaders | fraction=%.2f seed=%d | "
        "train=%d val=%d test=%d",
        subset_fraction, subset_seed,
        len(subset_indices), len(val_indices), len(test_dataset),
    )

    train_loader = DataLoader(
        Subset(full_train, subset_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_train_eval, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, subset_indices
