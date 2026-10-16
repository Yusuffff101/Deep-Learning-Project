"""
losses.py — Knowledge distillation loss functions for kd-lowdata.

Available methods
-----------------
ce                  Cross-entropy only (student baseline)
logit_kd            Hinton et al. 2015 — softened logit matching
attention_transfer  Zagoruyko & Komodakis 2017 — attention map matching

Extension guide
---------------
To add DKD, CRD, or any future method:
  1. Write a function with the signature::

         def dkd_loss(
             student_logits, teacher_logits, labels,
             student_features, teacher_features,
             **kwargs,
         ) -> torch.Tensor: ...

  2. Add it to LOSS_REGISTRY under the desired method name.
  3. Add any new hyperparameters to configs/default.yaml.

Nothing else needs to change — train.py resolves the loss by name from
the registry.

Interface contract
------------------
All loss functions must:
  - Accept (student_logits, teacher_logits, labels,
             student_features, teacher_features, **kwargs).
  - Return a scalar Tensor.
  - Be compatible with torch.cuda.amp (no explicit dtype casts needed;
    use F.log_softmax / F.softmax which are AMP-safe).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_finite(loss: torch.Tensor, name: str) -> None:
    """Raise RuntimeError if *loss* is not finite (debug aid)."""
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name} returned non-finite value: {loss.item()}")


# ── 1. Cross-entropy (baseline) ──────────────────────────────────────────────

def ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,   # unused, kept for uniform API
    labels: Optional[torch.Tensor] = None,
    student_features: Optional[List[torch.Tensor]] = None,   # unused
    teacher_features: Optional[List[torch.Tensor]] = None,   # unused
    **kwargs,
) -> torch.Tensor:
    """Standard cross-entropy loss.

    Parameters
    ----------
    student_logits:
        Raw model output of shape (N, C).
    labels:
        Ground-truth class indices of shape (N,).

    Returns
    -------
    Scalar CE loss.
    """
    if labels is None:
        raise ValueError("ce_loss requires `labels`.")
    return F.cross_entropy(student_logits, labels)


# ── 2. Logit-based KD (Hinton et al. 2015) ──────────────────────────────────

def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    student_features: Optional[List[torch.Tensor]] = None,   # unused
    teacher_features: Optional[List[torch.Tensor]] = None,   # unused
    T: float = 4.0,
    alpha: float = 0.9,
    **kwargs,
) -> torch.Tensor:
    """Hinton-style logit distillation.

    Loss = alpha * T² * KL(softmax(t_teacher/T) ‖ softmax(t_student/T))
           + (1 - alpha) * CE(t_student, labels)

    KL is computed as KL(teacher ‖ student) so the teacher distribution is
    the "target" and the student is the "prediction".

    Parameters
    ----------
    student_logits:
        Raw student output (N, C).
    teacher_logits:
        Raw teacher output (N, C). Teacher must be in eval / frozen.
    labels:
        Ground-truth class indices (N,).
    T:
        Temperature for softening distributions (default 4.0).
    alpha:
        Weight on the KL term; (1-alpha) on CE (default 0.9).

    Returns
    -------
    Scalar loss.
    """
    if teacher_logits is None:
        raise ValueError("logit_kd_loss requires `teacher_logits`.")
    if labels is None:
        raise ValueError("logit_kd_loss requires `labels`.")

    # Softened distributions
    log_p_s = F.log_softmax(student_logits / T, dim=1)   # student log-probs
    p_t = F.softmax(teacher_logits / T, dim=1)            # teacher probs

    # KL(teacher ‖ student) = sum_c p_t * (log p_t - log_p_s)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")

    ce = F.cross_entropy(student_logits, labels)

    loss = alpha * (T ** 2) * kl + (1.0 - alpha) * ce
    return loss


# ── 3. Attention Transfer (Zagoruyko & Komodakis 2017) ────────────────────────

def _attention_map(feat: torch.Tensor) -> torch.Tensor:
    """Compute spatial attention map from a feature tensor.

    Attention map = L2-normalised sum of squared activations across channels.

    Parameters
    ----------
    feat:
        Feature tensor of shape (N, C, H, W).

    Returns
    -------
    Attention map of shape (N, H*W), L2-normalised per sample.
    """
    # Sum over channels, then flatten spatial dims
    attn = feat.pow(2).mean(dim=1)           # (N, H, W)
    attn = attn.view(attn.size(0), -1)       # (N, H*W)
    # L2-normalise each sample's attention map
    attn = F.normalize(attn, p=2, dim=1)
    return attn


def attention_transfer_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    student_features: Optional[List[torch.Tensor]] = None,
    teacher_features: Optional[List[torch.Tensor]] = None,
    T: float = 4.0,
    alpha: float = 0.9,
    at_beta: float = 1000.0,
    **kwargs,
) -> torch.Tensor:
    """Attention-Transfer KD loss.

    Total loss = CE(student_logits, labels)
                 + at_beta * sum_stage ||A_s - A_t||²

    where A_s and A_t are L2-normalised spatial attention maps from
    matched intermediate stages.

    Parameters
    ----------
    student_logits, teacher_logits, labels:
        Standard outputs and ground truth.
    student_features:
        List of student feature maps [f1, f2, f3] from CIFARResNet.
    teacher_features:
        List of teacher feature maps [f1, f2, f3] from CIFARResNet.
        Must have the same number of stages as student_features.
    at_beta:
        Weight on the attention-transfer term (default 1000.0).

    Returns
    -------
    Scalar loss.
    """
    if labels is None:
        raise ValueError("attention_transfer_loss requires `labels`.")
    if student_features is None or teacher_features is None:
        raise ValueError(
            "attention_transfer_loss requires student_features and teacher_features."
        )
    if len(student_features) != len(teacher_features):
        raise ValueError(
            "student_features and teacher_features must have the same length."
        )

    ce = F.cross_entropy(student_logits, labels)

    at_loss = torch.tensor(0.0, device=student_logits.device)
    for s_feat, t_feat in zip(student_features, teacher_features):
        # Upsample student map to match teacher spatial size if needed
        # (happens when student/teacher have different architectures)
        s_attn = _attention_map(s_feat)
        t_attn = _attention_map(t_feat)

        if s_attn.shape != t_attn.shape:
            # Resize the smaller one to match the larger
            s_h = int(s_attn.shape[1] ** 0.5)
            t_h = int(t_attn.shape[1] ** 0.5)
            if s_h != t_h:
                # Reshape to spatial, interpolate, flatten back
                s_attn_2d = s_attn.view(s_attn.size(0), 1, s_h, s_h)
                s_attn_2d = F.interpolate(
                    s_attn_2d, size=(t_h, t_h), mode="bilinear", align_corners=False
                )
                s_attn = F.normalize(s_attn_2d.view(s_attn.size(0), -1), p=2, dim=1)

        at_loss = at_loss + (s_attn - t_attn).pow(2).mean()

    return ce + at_beta * at_loss


# ── Loss registry ─────────────────────────────────────────────────────────────

# Maps config method name → loss callable.
# The uniform signature is:
#   loss_fn(student_logits, teacher_logits, labels,
#            student_features, teacher_features, **kwargs) -> Tensor
#
# Add new methods here (e.g. "dkd": dkd_loss, "crd": crd_loss).
LOSS_REGISTRY: dict[str, Callable] = {
    "ce": ce_loss,
    "logit_kd": logit_kd_loss,
    "attention_transfer": attention_transfer_loss,
}


def get_loss_fn(method: str) -> Callable:
    """Retrieve a loss function from the registry by method name.

    Parameters
    ----------
    method:
        Method name, e.g. ``"ce"``, ``"logit_kd"``, ``"attention_transfer"``.

    Raises
    ------
    ValueError if *method* is not registered.
    """
    if method not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss method '{method}'. "
            f"Available: {list(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[method]
