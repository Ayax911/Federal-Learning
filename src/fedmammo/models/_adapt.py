"""Helpers to adapt torchvision backbones for grayscale input."""

from __future__ import annotations

import torch
from torch import nn


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a new Conv2d with ``in_channels`` channels, weights copied from ``conv``.

    For in_channels == conv.in_channels, ``conv`` is returned unchanged.
    For in_channels == 1 and conv.in_channels == 3, weights are averaged
    across the RGB axis (a standard approach in medical-imaging transfer
    learning — it preserves the response to luminance).
    For other reductions, weights are averaged across the first axis and
    then repeated.
    """
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        # Average across input channel dim, then tile/expand to new in_channels.
        averaged = conv.weight.mean(dim=1, keepdim=True)  # (out, 1, k, k)
        if in_channels == 1:
            new_conv.weight.copy_(averaged)
        else:
            new_conv.weight.copy_(averaged.repeat(1, in_channels, 1, 1))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


__all__ = ["adapt_first_conv"]
