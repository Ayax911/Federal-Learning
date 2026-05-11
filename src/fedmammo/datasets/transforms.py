"""Albumentations-based transform pipelines for mammography.

Two pipelines are produced by :func:`build_transforms`: a training transform
with stochastic augmentation, and an evaluation transform that only resizes
and normalizes. The augmentation set is conservative and clinically defensible
(no aggressive vertical flips on standard CC/MLO views unless explicitly
enabled).
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fedmammo.configs.schema import AugmentationConfig


def build_transforms(
    image_size: int,
    augmentation: AugmentationConfig,
    *,
    grayscale: bool = True,
) -> tuple[A.Compose, A.Compose]:
    """Build (train_transform, eval_transform).

    Args:
        image_size: Square resize target.
        augmentation: AugmentationConfig instance with knobs for the train pipeline.
        grayscale: If True, normalization uses 1-channel mean/std; else 3-channel.

    Returns:
        Tuple of Albumentations Compose objects. Both yield ``torch.Tensor``
        as their ``"image"`` output via :class:`ToTensorV2`.
    """
    if grayscale:
        mean: tuple[float, ...] = (augmentation.normalize_mean,)
        std: tuple[float, ...] = (augmentation.normalize_std,)
    else:
        mean = (augmentation.normalize_mean,) * 3
        std = (augmentation.normalize_std,) * 3

    train_ops: list[Any] = [A.Resize(image_size, image_size)]
    if augmentation.horizontal_flip:
        train_ops.append(A.HorizontalFlip(p=0.5))
    if augmentation.vertical_flip:
        train_ops.append(A.VerticalFlip(p=0.5))
    if augmentation.rotate_limit > 0:
        train_ops.append(
            A.Rotate(limit=augmentation.rotate_limit, border_mode=0, p=0.5)
        )
    if augmentation.brightness_contrast:
        train_ops.append(
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        )
    if augmentation.elastic:
        train_ops.append(A.ElasticTransform(alpha=20, sigma=5, p=0.3))
    train_ops.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    eval_ops: list[Any] = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(train_ops), A.Compose(eval_ops)


__all__ = ["build_transforms"]
