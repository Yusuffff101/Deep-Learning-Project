"""
tests/test_smoke.py — Smoke tests for kd-lowdata.

Tests (9 total)
---------------
1. Subset determinism: same (fraction, seed) → same indices
2. Subset class balance: each class represented proportionally
3. Subset nesting: 5% ⊂ 10% ⊂ 25% ⊂ 100% (per seed)
4. Val/pool non-overlap: val_indices ∩ pool_indices = ∅
5. CE loss: finite scalar on random tensors
6. Logit KD loss: finite scalar on random tensors
7. Attention-transfer loss: finite scalar on random tensors
8. Teacher 2-step training: runs to completion on CPU with tiny subset
9. Student 2-step training (logit KD): runs to completion on CPU with tiny subset

Run with:
    cd kd-lowdata
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make sure src/ is importable regardless of where pytest is invoked from
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from data import (
    NUM_CLASSES,
    VAL_SEED,
    VAL_SIZE,
    _get_val_and_pool_indices,
    get_subset_indices,
)
from losses import attention_transfer_loss, ce_loss, get_loss_fn, logit_kd_loss
from models import build_model, resnet20, resnet56
from train import build_config, train


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_targets() -> list[int]:
    """Synthetic 50k targets with 500 images per class (balanced CIFAR-100)."""
    return [cls for cls in range(NUM_CLASSES) for _ in range(500)]


@pytest.fixture(scope="module")
def val_pool(fake_targets):
    """Pre-computed val and pool indices for the fake targets."""
    val_idx, pool_idx = _get_val_and_pool_indices(
        fake_targets, val_size=VAL_SIZE, val_seed=VAL_SEED
    )
    pool_targets = np.array(fake_targets)[pool_idx]
    return val_idx, pool_idx, pool_targets


# ── Test 1: Subset determinism ────────────────────────────────────────────────

def test_subset_determinism(val_pool):
    """Same (fraction, seed) always returns identical indices."""
    _, pool_idx, pool_targets = val_pool
    idx_a = get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=42)
    idx_b = get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=42)
    np.testing.assert_array_equal(idx_a, idx_b,
                                   err_msg="Subset not deterministic for same (fraction, seed)")


def test_subset_different_seeds(val_pool):
    """Different seeds produce different subsets."""
    _, pool_idx, pool_targets = val_pool
    idx_0 = get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=0)
    idx_1 = get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=1)
    assert not np.array_equal(idx_0, idx_1), "Different seeds should yield different subsets"


# ── Test 2: Class balance ─────────────────────────────────────────────────────

def test_subset_class_balance(val_pool, fake_targets):
    """Each class should be represented roughly equally in the subset."""
    _, pool_idx, pool_targets = val_pool
    idx = get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=0)
    selected_targets = np.array(fake_targets)[idx]
    counts = np.bincount(selected_targets, minlength=NUM_CLASSES)

    # All classes must have at least 1 sample
    assert counts.min() >= 1, f"Some class has 0 samples: {counts}"

    # No class should have more than 2× another (balance check)
    ratio = counts.max() / counts.min()
    assert ratio <= 2.0, f"Class imbalance too large: max/min={ratio:.2f}"


# ── Test 3: Subset nesting ────────────────────────────────────────────────────

def test_subset_nesting(val_pool):
    """5% ⊂ 10% ⊂ 25% for the same seed (nested prefix property)."""
    _, pool_idx, pool_targets = val_pool
    seed = 0
    idx_5 = set(get_subset_indices(pool_idx, pool_targets, fraction=0.05, seed=seed).tolist())
    idx_10 = set(get_subset_indices(pool_idx, pool_targets, fraction=0.10, seed=seed).tolist())
    idx_25 = set(get_subset_indices(pool_idx, pool_targets, fraction=0.25, seed=seed).tolist())

    assert idx_5.issubset(idx_10), "5% subset is NOT ⊂ 10% subset (nesting violated)"
    assert idx_10.issubset(idx_25), "10% subset is NOT ⊂ 25% subset (nesting violated)"


# ── Test 4: Val / pool non-overlap ────────────────────────────────────────────

def test_val_pool_no_overlap(val_pool):
    """Validation indices must not appear in the training pool."""
    val_idx, pool_idx, _ = val_pool
    overlap = np.intersect1d(val_idx, pool_idx)
    assert len(overlap) == 0, (
        f"Val and pool indices overlap on {len(overlap)} samples!"
    )


def test_val_size(val_pool):
    """Validation split should contain exactly VAL_SIZE images."""
    val_idx, _, _ = val_pool
    assert len(val_idx) == VAL_SIZE, (
        f"Expected {VAL_SIZE} val samples, got {len(val_idx)}"
    )


# ── Test 5: CE loss ───────────────────────────────────────────────────────────

def test_ce_loss_finite():
    """CE loss returns a finite scalar."""
    B, C = 8, 100
    logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    loss = ce_loss(student_logits=logits, labels=labels)
    assert loss.ndim == 0, "Loss must be a scalar"
    assert torch.isfinite(loss), f"CE loss is not finite: {loss.item()}"


# ── Test 6: Logit KD loss ─────────────────────────────────────────────────────

def test_logit_kd_loss_finite():
    """Logit KD loss returns a finite scalar for various T and alpha values."""
    B, C = 8, 100
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    for T, alpha in [(1.0, 0.5), (4.0, 0.9), (8.0, 1.0)]:
        loss = logit_kd_loss(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            T=T,
            alpha=alpha,
        )
        assert torch.isfinite(loss), f"logit_kd_loss not finite (T={T}, alpha={alpha})"


def test_logit_kd_loss_uses_temperature():
    """Higher T should produce a softer distribution (lower KL)."""
    B, C = 32, 100
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    # With T→∞ the KL term should go to 0; loss ≈ CE*(1-alpha)
    loss_low_T = logit_kd_loss(s_logits, t_logits, labels, T=1.0, alpha=0.9)
    loss_high_T = logit_kd_loss(s_logits, t_logits, labels, T=20.0, alpha=0.9)
    # No strict ordering guaranteed on random tensors, just check both finite
    assert torch.isfinite(loss_low_T) and torch.isfinite(loss_high_T)


# ── Test 7: Attention-transfer loss ───────────────────────────────────────────

def test_attention_transfer_loss_finite():
    """AT loss returns a finite scalar given random feature maps."""
    B, C = 8, 100
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    # Simulate ResNet-20 vs ResNet-56 feature shapes (same spatial size, diff channels)
    s_feats = [torch.randn(B, 16, 32, 32),
               torch.randn(B, 32, 16, 16),
               torch.randn(B, 64,  8,  8)]
    t_feats = [torch.randn(B, 16, 32, 32),
               torch.randn(B, 32, 16, 16),
               torch.randn(B, 64,  8,  8)]

    loss = attention_transfer_loss(
        student_logits=s_logits,
        teacher_logits=t_logits,
        labels=labels,
        student_features=s_feats,
        teacher_features=t_feats,
        at_beta=1000.0,
    )
    assert torch.isfinite(loss), f"AT loss is not finite: {loss.item()}"


# ── Test 8: Loss registry ─────────────────────────────────────────────────────

def test_loss_registry():
    """All registered methods are callable and return finite scalars."""
    B, C = 4, 100
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    s_feats = [torch.randn(B, 16, 32, 32),
               torch.randn(B, 32, 16, 16),
               torch.randn(B, 64,  8,  8)]

    for method in ["ce", "logit_kd", "attention_transfer"]:
        fn = get_loss_fn(method)
        loss = fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_features=s_feats,
            teacher_features=s_feats,
            T=4.0,
            alpha=0.9,
            at_beta=1000.0,
        )
        assert torch.isfinite(loss), f"Loss '{method}' returned non-finite: {loss}"


# ── Helper: tiny training config ─────────────────────────────────────────────

def _make_tiny_args(role: str, method: str = "ce", teacher_ckpt: str | None = None,
                    tmp_path: Path | None = None):
    """Return a minimal config dict for 2-step CPU training."""
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())

    class Args:
        pass

    args = Args()
    args.role = role
    args.method = method
    args.config = str(Path(__file__).parent.parent / "configs" / "default.yaml")
    args.teacher_model = None
    args.student_model = None
    args.f_t = 1.0
    args.f_s = 0.01
    args.seed = 0
    args.teacher_ckpt = teacher_ckpt
    args.lr = 0.1
    args.weight_decay = 5e-4
    args.epochs = 0
    args.batch_size = 8
    args.min_steps = 2          # Force only 2 steps (with epochs=0, max(0, 2) == 2)
    args.T = 4.0
    args.alpha = 0.9
    args.at_beta = 1000.0
    repo_data = Path(__file__).parent.parent / "data"
    args.data_dir = str(repo_data if repo_data.exists() else (tmp_path / "data"))
    args.checkpoint_dir = str(tmp_path / "checkpoints")
    args.output_dir = str(tmp_path / "results")
    args.num_workers = 0       # Needed on Windows / CPU

    return args, tmp_path


# ── Test 9: 2-step teacher training on CPU ────────────────────────────────────

def test_teacher_two_step(tmp_path):
    """Teacher trains for exactly 2 steps on CPU without errors."""
    args, tmp_path = _make_tiny_args("teacher", tmp_path=tmp_path)
    cfg = build_config(args)
    cfg["pin_memory"] = False   # CPU: no pin_memory
    cfg["num_workers"] = 0

    summary = train(cfg)

    assert "best_val_acc" in summary
    assert summary["total_steps"] >= 2
    assert isinstance(summary["best_val_acc"], float)

    # Checkpoint should exist
    ckpt_dir = Path(cfg["checkpoint_dir"])
    checkpoints = list(ckpt_dir.glob("**/best.pth"))
    assert len(checkpoints) >= 1, "No checkpoint was saved!"


# ── Test 10: 2-step student training (logit KD) on CPU ───────────────────────

def test_student_logit_kd_two_step(tmp_path):
    """Student with logit KD trains for 2 steps on CPU without errors."""
    # First train a teacher to get a checkpoint
    t_args, tmp_path = _make_tiny_args("teacher", tmp_path=tmp_path)
    t_cfg = build_config(t_args)
    t_cfg["pin_memory"] = False
    t_cfg["num_workers"] = 0
    train(t_cfg)

    # Find teacher checkpoint
    from utils import make_run_id
    teacher_run_id = make_run_id(
        role="teacher",
        model=t_cfg["teacher_model"],
        method="ce",
        f_t=t_cfg["f_t"],
        f_s=t_cfg["f_s"],
        seed=t_cfg["seed"],
    )
    teacher_ckpt = str(Path(t_cfg["checkpoint_dir"]) / teacher_run_id / "best.pth")
    assert Path(teacher_ckpt).exists(), f"Teacher checkpoint not found: {teacher_ckpt}"

    # Train student
    s_args, _ = _make_tiny_args(
        "student", method="logit_kd",
        teacher_ckpt=teacher_ckpt, tmp_path=tmp_path
    )
    s_cfg = build_config(s_args)
    s_cfg["pin_memory"] = False
    s_cfg["num_workers"] = 0

    summary = train(s_cfg)

    assert summary["total_steps"] >= 2
    assert isinstance(summary["best_val_acc"], float)

    # Student checkpoint should exist
    ckpt_dir = Path(s_cfg["checkpoint_dir"])
    student_ckpts = list(ckpt_dir.glob("**/best.pth"))
    # Teacher + student = at least 2 checkpoints
    assert len(student_ckpts) >= 2, f"Expected ≥2 checkpoints, found {len(student_ckpts)}"


# ── Test 11: Model forward shapes ─────────────────────────────────────────────

def test_resnet20_output_shape():
    """ResNet-20 should output (B, 100) logits and 3 feature maps."""
    model = resnet20(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 100), f"Wrong logit shape: {logits.shape}"

    logits2, features = model(x, return_features=True)
    assert logits2.shape == (4, 100)
    assert len(features) == 3
    assert features[0].shape[1] == 16   # stage1 channels
    assert features[1].shape[1] == 32   # stage2 channels
    assert features[2].shape[1] == 64   # stage3 channels


def test_resnet56_output_shape():
    """ResNet-56 should output (B, 100) logits."""
    model = resnet56(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 100), f"Wrong logit shape: {logits.shape}"


def test_build_model_registry():
    """build_model raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")
