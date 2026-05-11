"""ResNet18 classifier for binary mammography classification."""

from __future__ import annotations

import torch
from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models._adapt import adapt_first_conv
from fedmammo.models.factory import register_model


class ResNet18Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.resnet18``.

    - First conv adapted to ``in_channels`` (default 1 for grayscale mammograms).
    - Final FC replaced with ``Dropout(p) -> Linear(in_features, num_classes)``.

    The wrapper keeps the underlying ResNet attribute reachable as
    ``self.backbone`` for downstream inspection or layer-wise LR schedules.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Local import keeps torchvision optional at module load time.
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
        backbone = resnet18(weights=weights)

        if cfg.in_channels != 3:
            backbone.conv1 = adapt_first_conv(backbone.conv1, cfg.in_channels)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features, cfg.num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_model("resnet18")
def _build_resnet18(cfg: ModelConfig) -> nn.Module:
    return ResNet18Classifier(cfg)


__all__ = ["ResNet18Classifier"]
