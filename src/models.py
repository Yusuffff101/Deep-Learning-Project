"""
models.py — CIFAR-style ResNet-20 and ResNet-56 for kd-lowdata.

Architecture follows He et al. (2016) "Deep Residual Learning for Image
Recognition", CIFAR variant:
  - Initial conv 3×3, stride 1 (not the ImageNet 7×7)
  - Three stages of n residual blocks (n=3 → ResNet-20, n=9 → ResNet-56)
  - Widths: 16 → 32 → 64
  - Global average pooling → FC(100)

The forward method accepts an optional `return_features` flag.
When True, it returns (logits, features) where features is a list of
activation tensors after each stage — used by attention-transfer and
FitNets KD losses.

Stages exposed for feature KD:
  features[0]: after stage 1 (shape: B × 16 × 32 × 32)
  features[1]: after stage 2 (shape: B × 32 × 16 × 16)
  features[2]: after stage 3 (shape: B × 64 ×  8 ×  8)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic residual block (no bottleneck) ─────────────────────────────────────

class BasicBlock(nn.Module):
    """Two-layer residual block used in CIFAR ResNets (no bottleneck)."""

    expansion: int = 1  # output channels == in_planes * expansion

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection: identity when dimensions match,
        # 1×1 conv when spatial size or channels change.
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ── CIFAR ResNet ─────────────────────────────────────────────────────────────

class CIFARResNet(nn.Module):
    """CIFAR-style ResNet with three stages of BasicBlocks.

    Parameters
    ----------
    num_blocks:
        Number of BasicBlocks per stage. ResNet-20 → [3,3,3],
        ResNet-56 → [9,9,9].
    num_classes:
        Output dimension of the final FC layer.
    """

    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        self.in_planes = 16

        # Initial conv: 3×3, stride 1 (CIFAR-specific — NOT the ImageNet stem)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three residual stages
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        # Weight initialisation (matches He et al.)
        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build one residual stage."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, 3, 32, 32).
        return_features:
            If True, also return a list of intermediate feature maps
            [feat1, feat2, feat3] — one per residual stage.

        Returns
        -------
        logits:
            Class logits of shape (B, num_classes).
        features (only when return_features=True):
            List of 3 tensors: after stage1, stage2, stage3.
        """
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual stages
        f1 = self.layer1(out)   # B × 16 × 32 × 32
        f2 = self.layer2(f1)    # B × 32 × 16 × 16
        f3 = self.layer3(f2)    # B × 64 ×  8 ×  8

        # Head
        out = self.avgpool(f3)
        out = torch.flatten(out, 1)
        logits = self.fc(out)

        if return_features:
            return logits, [f1, f2, f3]
        return logits


# ── Public constructors ───────────────────────────────────────────────────────

def resnet20(num_classes: int = 100) -> CIFARResNet:
    """CIFAR ResNet-20 (6n+2 = 20 layers, n=3 blocks per stage)."""
    return CIFARResNet(num_blocks=[3, 3, 3], num_classes=num_classes)


def resnet56(num_classes: int = 100) -> CIFARResNet:
    """CIFAR ResNet-56 (6n+2 = 56 layers, n=9 blocks per stage)."""
    return CIFARResNet(num_blocks=[9, 9, 9], num_classes=num_classes)


# Registry: map config string → constructor
MODEL_REGISTRY: dict[str, type[CIFARResNet]] = {
    "resnet20": resnet20,
    "resnet56": resnet56,
}


def build_model(name: str, num_classes: int = 100) -> CIFARResNet:
    """Instantiate a model by registry name.

    Parameters
    ----------
    name:
        One of the keys in MODEL_REGISTRY (e.g. ``"resnet20"``).
    num_classes:
        Number of output classes.

    Raises
    ------
    ValueError if *name* is not found in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)
