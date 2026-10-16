"""
utils.py — Shared utilities for kd-lowdata.

Contents:
  - set_seeds: reproducible seed setting across all libraries
  - load_yaml_config: YAML loader returning a dict
  - save_checkpoint / load_checkpoint: thin wrappers around torch.save/load
  - AverageMeter: running mean tracker for losses / accuracies
  - get_device: return torch.device, honouring CUDA availability
  - setup_logging: basic logging configuration
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Sets Python, NumPy, and PyTorch (CPU + CUDA) seeds.
    Also enables PyTorch deterministic mode where practical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training; acceptable for research correctness)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow-merge *override* into *base*, returning a new dict.

    Override wins for every key present in both dicts.
    """
    merged = base.copy()
    merged.update(override)
    return merged


def save_json(obj: Any, path: str | Path) -> None:
    """Serialize *obj* to JSON at *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> None:
    """Save *state* dict to *path* using torch.save.

    Creates parent directories automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load and return a checkpoint saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        File path to the checkpoint.
    device:
        If provided, maps storage to this device (useful for CPU-only machines).
    """
    if device is None:
        device = get_device()
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    """Compute and store a running average of a scalar value.

    Usage::

        meter = AverageMeter("loss")
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update with a new value averaged over *n* samples."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f})"


def accuracy_topk(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Return top-k accuracy percentages for each k in *topk*.

    Parameters
    ----------
    output:
        Model logits of shape (N, C).
    target:
        Ground-truth class indices of shape (N,).
    topk:
        Tuple of k values to compute accuracy for.

    Returns
    -------
    List of accuracy percentages (0–100), one per k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Run-ID helper
# ---------------------------------------------------------------------------

def make_run_id(
    role: str,
    model: str,
    method: str,
    f_t: float,
    f_s: float,
    seed: int,
) -> str:
    """Return a short, filesystem-safe run identifier.

    Example: ``teacher_resnet56_ce_fT1.00_fS1.00_s0``
    """
    return (
        f"{role}_{model}_{method}"
        f"_fT{f_t:.2f}_fS{f_s:.2f}_s{seed}"
    )
