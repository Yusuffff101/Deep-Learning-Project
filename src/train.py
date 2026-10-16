"""
train.py — Teacher and student training for kd-lowdata.

Usage
-----
# Train a teacher (ResNet-56) on 100% of the data:
python src/train.py --role teacher --f_t 1.0 --seed 0

# Train a student (ResNet-20) with logit KD from a teacher checkpoint:
python src/train.py --role student --method logit_kd \\
    --f_s 0.05 --f_t 1.0 --seed 0 \\
    --teacher_ckpt checkpoints/teacher_resnet56_ce_fT1.00_fS1.00_s0/best.pth

Key design decisions
---------------------
- Training budget: max(epochs * steps_per_epoch, min_steps).
  This prevents tiny subsets from being severely under-trained.
- Best checkpoint is selected by VALIDATION accuracy only.
  The test set is never touched during training.
- Teacher is loaded once, frozen (eval mode, no grad), and receives the
  EXACT SAME augmented batch as the student.
- Mixed precision (torch.cuda.amp) is used on CUDA; plain FP32 on CPU.
- All random seeds are set at the start of every run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data import build_dataloaders
from losses import get_loss_fn
from models import build_model
from utils import (
    AverageMeter,
    accuracy_topk,
    get_device,
    load_checkpoint,
    load_yaml_config,
    make_run_id,
    save_checkpoint,
    save_json,
    set_seeds,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train teacher or student model for kd-lowdata."
    )
    # Role
    p.add_argument("--role", choices=["teacher", "student"], required=True,
                   help="Whether to train the teacher or student model.")
    p.add_argument("--method", default="ce",
                   choices=["ce", "logit_kd", "attention_transfer"],
                   help="Loss method. For teacher, always CE regardless of this flag.")

    # Config
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file.")

    # Model
    p.add_argument("--teacher_model", default=None,
                   help="Override teacher model name (default from config).")
    p.add_argument("--student_model", default=None,
                   help="Override student model name (default from config).")

    # Data fractions
    p.add_argument("--f_t", type=float, default=None,
                   help="Teacher data fraction (0,1]. Overrides config.")
    p.add_argument("--f_s", type=float, default=None,
                   help="Student data fraction (0,1]. Overrides config.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. Overrides config.")

    # Teacher checkpoint (student training only)
    p.add_argument("--teacher_ckpt", default=None,
                   help="Path to teacher checkpoint .pth file.")

    # Training hyperparams (override config)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--min_steps", type=int, default=None)
    p.add_argument("--T", type=float, default=None, help="KD temperature.")
    p.add_argument("--alpha", type=float, default=None, help="KD alpha.")
    p.add_argument("--at_beta", type=float, default=None,
                   help="Attention-transfer beta weight.")

    # Paths
    p.add_argument("--data_dir", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--output_dir", default=None)

    # Misc
    p.add_argument("--num_workers", type=int, default=None)

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides. CLI always wins."""
    cfg = load_yaml_config(args.config)

    # Apply CLI overrides (only if provided)
    overrides = {
        "role": args.role,
        "method": args.method if args.role == "student" else "ce",
        "teacher_model": args.teacher_model or cfg.get("teacher_model", "resnet56"),
        "student_model": args.student_model or cfg.get("student_model", "resnet20"),
        "f_t": args.f_t if args.f_t is not None else cfg.get("teacher_fractions", [1.0])[0],
        "f_s": args.f_s if args.f_s is not None else cfg.get("student_fractions", [1.0])[0],
        "seed": args.seed if args.seed is not None else cfg.get("seeds", [0])[0],
        "teacher_ckpt": args.teacher_ckpt,
        "lr": args.lr if args.lr is not None else cfg.get("lr", 0.1),
        "weight_decay": args.weight_decay if args.weight_decay is not None else cfg.get("weight_decay", 5e-4),
        "epochs": args.epochs if args.epochs is not None else cfg.get("epochs", 200),
        "batch_size": args.batch_size if args.batch_size is not None else cfg.get("batch_size", 128),
        "min_steps": args.min_steps if args.min_steps is not None else cfg.get("min_steps", 30000),
        "T": args.T if args.T is not None else cfg.get("temperature", 4.0),
        "alpha": args.alpha if args.alpha is not None else cfg.get("alpha", 0.9),
        "at_beta": args.at_beta if args.at_beta is not None else cfg.get("at_beta", 1000.0),
        "data_dir": args.data_dir or cfg.get("data_dir", "./data"),
        "checkpoint_dir": args.checkpoint_dir or cfg.get("checkpoint_dir", "./checkpoints"),
        "output_dir": args.output_dir or cfg.get("output_dir", "./results"),
        "num_workers": args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4),
        "num_classes": cfg.get("num_classes", 100),
        "ece_bins": cfg.get("ece_bins", 15),
        "log_interval": cfg.get("log_interval", 50),
        "pin_memory": cfg.get("pin_memory", True),
    }
    return overrides


# ── Optimizer & scheduler ────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=cfg["weight_decay"],
        nesterov=True,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine annealing over the full training budget (in steps)."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )


# ── One training step ────────────────────────────────────────────────────────

def train_step(
    *,
    model: nn.Module,
    teacher: Optional[nn.Module],
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    loss_fn,
    cfg: dict,
    device: torch.device,
    needs_features: bool,
) -> float:
    """Execute a single forward-backward-update step.

    Returns
    -------
    Scalar loss value (float).
    """
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    model.train()

    use_amp = scaler is not None

    with autocast(device_type=device.type, enabled=use_amp):
        # Forward: student (and optionally teacher)
        if needs_features:
            s_logits, s_features = model(images, return_features=True)
        else:
            s_logits = model(images)
            s_features = None

        # Teacher inference (frozen, no grad)
        t_logits = None
        t_features = None
        if teacher is not None:
            with torch.no_grad():
                if needs_features:
                    t_logits, t_features = teacher(images, return_features=True)
                else:
                    t_logits = teacher(images)

        loss = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_features=s_features,
            teacher_features=t_features,
            T=cfg["T"],
            alpha=cfg["alpha"],
            at_beta=cfg["at_beta"],
        )

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    scheduler.step()
    return loss.item()


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Return top-1 validation accuracy (%)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ── Main training loop ───────────────────────────────────────────────────────

def train(cfg: dict) -> dict:
    """Run full training according to *cfg*.

    Returns
    -------
    A dict with keys: best_val_acc, total_steps, wall_time_s, run_id.
    """
    set_seeds(cfg["seed"])
    device = get_device()
    logger.info("Device: %s", device)

    # Determine which fraction governs the training data
    fraction = cfg["f_t"] if cfg["role"] == "teacher" else cfg["f_s"]
    model_name = (
        cfg["teacher_model"] if cfg["role"] == "teacher" else cfg["student_model"]
    )

    # Build run ID and output directory
    run_id = make_run_id(
        role=cfg["role"],
        model=model_name,
        method=cfg["method"],
        f_t=cfg["f_t"],
        f_s=cfg["f_s"],
        seed=cfg["seed"],
    )
    ckpt_dir = Path(cfg["checkpoint_dir"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run: %s", run_id)

    # Save config for this run
    save_json(cfg, ckpt_dir / "config.json")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _test_loader, _subset_idx = build_dataloaders(
        data_dir=cfg["data_dir"],
        subset_fraction=fraction,
        subset_seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"] and device.type == "cuda",
    )

    steps_per_epoch = len(train_loader)
    total_steps = max(cfg["epochs"] * steps_per_epoch, cfg["min_steps"])
    total_epochs = math.ceil(total_steps / max(steps_per_epoch, 1))
    logger.info(
        "Training budget: %d steps (%d epochs × %d steps/epoch, min_steps=%d)",
        total_steps, total_epochs, steps_per_epoch, cfg["min_steps"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(model_name, num_classes=cfg["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s | %.2fM parameters", model_name, n_params / 1e6)

    # ── Teacher (student role only) ──────────────────────────────────────────
    teacher: Optional[nn.Module] = None
    needs_features = cfg["method"] == "attention_transfer"

    if cfg["role"] == "student":
        if cfg["teacher_ckpt"] is None:
            raise ValueError("Student training requires --teacher_ckpt.")
        ckpt = load_checkpoint(cfg["teacher_ckpt"], device=device)
        teacher = build_model(
            cfg["teacher_model"], num_classes=cfg["num_classes"]
        ).to(device)
        teacher.load_state_dict(ckpt["model_state"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        logger.info("Teacher loaded from %s", cfg["teacher_ckpt"])

    # ── Loss function ─────────────────────────────────────────────────────────
    loss_fn = get_loss_fn(cfg["method"])

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, total_steps)

    # Mixed precision only on CUDA
    scaler: Optional[GradScaler] = GradScaler(device="cuda") if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc: float = 0.0
    step = 0
    loss_meter = AverageMeter("loss")
    t_start = time.time()

    epoch = 0
    pbar = tqdm(total=total_steps, desc=f"[{run_id}]", unit="step")

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            loss_val = train_step(
                model=model,
                teacher=teacher,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_fn=loss_fn,
                cfg=cfg,
                device=device,
                needs_features=needs_features,
            )
            loss_meter.update(loss_val, n=batch[0].size(0))
            step += 1
            pbar.update(1)

            if step % cfg["log_interval"] == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}", lr=f"{lr_now:.5f}"
                )
                loss_meter.reset()

        # Validate at end of each epoch
        val_acc = evaluate(model, val_loader, device)
        logger.info("Epoch %d | step %d | val_acc=%.2f%%", epoch, step, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "step": step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                ckpt_dir / "best.pth",
            )
            logger.info("  ↳ New best val_acc=%.2f%% — checkpoint saved.", val_acc)

    pbar.close()
    wall_time = time.time() - t_start

    logger.info(
        "Training complete | best_val_acc=%.2f%% | steps=%d | time=%.1fs",
        best_val_acc, step, wall_time,
    )

    # Save final summary
    summary = {
        "run_id": run_id,
        "best_val_acc": best_val_acc,
        "total_steps": step,
        "wall_time_s": wall_time,
        "cfg": cfg,
    }
    save_json(summary, ckpt_dir / "summary.json")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    setup_logging()
    args = parse_args(argv)
    cfg = build_config(args)
    logger.info("Config: %s", json.dumps(cfg, indent=2, default=str))
    train(cfg)


if __name__ == "__main__":
    main()
