"""
evaluate.py — Test-set evaluation for kd-lowdata.

Computes and saves:
  - Top-1 accuracy
  - Student-teacher agreement (fidelity): fraction of test images where
    student and teacher make the same top-1 prediction
  - ECE (Expected Calibration Error, 15 bins by default)
  - Per-class accuracy
  - Raw predictions and probabilities (for offline re-analysis)

Usage
-----
python src/evaluate.py \\
    --student_ckpt checkpoints/<run_id>/best.pth \\
    --teacher_ckpt checkpoints/<teacher_run_id>/best.pth \\
    --output_dir results/eval/<run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data import build_dataloaders
from models import build_model
from utils import (
    get_device,
    load_checkpoint,
    load_yaml_config,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ── ECE computation ─────────────────────────────────────────────────────────

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error using equal-width confidence bins.

    Parameters
    ----------
    probs:
        Softmax probabilities of shape (N, C).
    labels:
        Ground-truth class indices of shape (N,).
    n_bins:
        Number of equal-width confidence bins (default 15).

    Returns
    -------
    ECE as a float in [0, 1].
    """
    confidences = probs.max(axis=1)         # (N,) max probability
    predictions = probs.argmax(axis=1)      # (N,) predicted class
    accuracies = predictions == labels       # (N,) boolean

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / n)


# ── Per-class accuracy ────────────────────────────────────────────────────────

def compute_per_class_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> dict[int, float]:
    """Return per-class top-1 accuracy as a dict {class_id: accuracy}."""
    per_class = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            per_class[cls] = float("nan")
        else:
            per_class[cls] = float((predictions[mask] == cls).mean())
    return per_class


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run *model* on all batches in *loader*.

    Returns
    -------
    all_probs:   (N, C) softmax probabilities
    all_preds:   (N,)   top-1 predicted class indices
    all_labels:  (N,)   ground-truth class indices
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ── Main evaluation function ─────────────────────────────────────────────────

def evaluate_checkpoint(
    student_ckpt_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    teacher_ckpt_path: Optional[str | Path] = None,
    num_classes: int = 100,
    batch_size: int = 128,
    num_workers: int = 4,
    ece_bins: int = 15,
    config_overrides: Optional[dict] = None,
) -> dict:
    """Evaluate a student checkpoint on the CIFAR-100 test set.

    Parameters
    ----------
    student_ckpt_path:
        Path to the student (or teacher) checkpoint .pth file.
    data_dir:
        Root directory where CIFAR-100 is stored.
    output_dir:
        Directory where evaluation outputs will be saved.
    teacher_ckpt_path:
        Optional teacher checkpoint for computing agreement/fidelity.
    num_classes:
        Number of output classes.
    batch_size, num_workers, ece_bins:
        Evaluation parameters.

    Returns
    -------
    Dict with metrics: top1_acc, ece, agreement (if teacher provided),
    and paths to saved outputs.
    """
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student checkpoint
    ckpt = load_checkpoint(student_ckpt_path, device=device)
    cfg = ckpt.get("cfg", config_overrides or {})

    student_model_name = cfg.get("student_model", cfg.get("teacher_model", "resnet20"))
    student = build_model(student_model_name, num_classes=num_classes).to(device)
    student.load_state_dict(ckpt["model_state"])
    logger.info("Student loaded: %s from %s", student_model_name, student_ckpt_path)

    # Build test loader (fraction=1.0 to use full pool, but we only use test set)
    _train, _val, test_loader, _ = build_dataloaders(
        data_dir=data_dir,
        subset_fraction=1.0,
        subset_seed=0,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Student inference
    s_probs, s_preds, labels = run_inference(student, test_loader, device)

    top1_acc = float((s_preds == labels).mean() * 100.0)
    ece = compute_ece(s_probs, labels, n_bins=ece_bins)
    per_class = compute_per_class_accuracy(s_preds, labels, num_classes=num_classes)

    logger.info("Test top-1 acc: %.2f%% | ECE: %.4f", top1_acc, ece)

    # Agreement with teacher (if provided)
    agreement: Optional[float] = None
    if teacher_ckpt_path is not None:
        t_ckpt = load_checkpoint(teacher_ckpt_path, device=device)
        t_cfg = t_ckpt.get("cfg", {})
        teacher_model_name = t_cfg.get("teacher_model", "resnet56")
        teacher = build_model(teacher_model_name, num_classes=num_classes).to(device)
        teacher.load_state_dict(t_ckpt["model_state"])
        logger.info("Teacher loaded: %s from %s", teacher_model_name, teacher_ckpt_path)

        t_probs, t_preds, _ = run_inference(teacher, test_loader, device)
        agreement = float((s_preds == t_preds).mean() * 100.0)
        logger.info("Student-teacher agreement: %.2f%%", agreement)

        # Save teacher predictions too
        np.save(output_dir / "teacher_probs.npy", t_probs)
        np.save(output_dir / "teacher_preds.npy", t_preds)

    # Save student predictions and probabilities
    np.save(output_dir / "student_probs.npy", s_probs)
    np.save(output_dir / "student_preds.npy", s_preds)
    np.save(output_dir / "labels.npy", labels)

    results = {
        "top1_acc": top1_acc,
        "ece": ece,
        "agreement": agreement,
        "per_class_accuracy": per_class,
        "student_ckpt": str(student_ckpt_path),
        "teacher_ckpt": str(teacher_ckpt_path) if teacher_ckpt_path else None,
        "num_test_samples": int(len(labels)),
    }
    save_json(results, output_dir / "metrics.json")
    logger.info("Evaluation saved to %s", output_dir)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a kd-lowdata checkpoint on CIFAR-100 test set.")
    p.add_argument("--student_ckpt", required=True,
                   help="Path to student (or teacher) best.pth checkpoint.")
    p.add_argument("--teacher_ckpt", default=None,
                   help="Path to teacher best.pth (for agreement metric).")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write evaluation outputs.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    setup_logging()
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    evaluate_checkpoint(
        student_ckpt_path=args.student_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_ckpt_path=args.teacher_ckpt,
        num_classes=cfg.get("num_classes", 100),
        batch_size=args.batch_size or cfg.get("batch_size", 128),
        num_workers=args.num_workers or cfg.get("num_workers", 4),
        ece_bins=cfg.get("ece_bins", 15),
    )


if __name__ == "__main__":
    main()
