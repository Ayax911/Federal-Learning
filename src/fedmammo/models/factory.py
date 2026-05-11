"""Model factory + registry.

Models register themselves with :func:`register_model` at import time, and
:func:`build_model` is the single switch point used by training/FL code.

Adding a model:

    @register_model("my_arch")
    def _build_my_arch(cfg: ModelConfig) -> nn.Module:
        ...
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from fedmammo.configs.schema import ModelConfig

_BuildFn = Callable[[ModelConfig], nn.Module]
_REGISTRY: dict[str, _BuildFn] = {}


def register_model(name: str) -> Callable[[_BuildFn], _BuildFn]:
    """Decorator that registers a builder under ``name``."""

    def _decorator(fn: _BuildFn) -> _BuildFn:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model name already registered: {name!r}")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a model from its config."""
    key = cfg.name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: {cfg.name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    model = _REGISTRY[key](cfg)
    return model


def list_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


__all__ = ["build_model", "register_model", "list_models"]
