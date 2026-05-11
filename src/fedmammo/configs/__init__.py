"""Configuration system for fedmammo.

Exposes the typed config dataclasses and a YAML loader.
"""

from fedmammo.configs.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    PartitioningConfig,
    StrategyConfig,
    TrainingConfig,
)
from fedmammo.configs.loader import load_config, save_config

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FederatedConfig",
    "ModelConfig",
    "PartitioningConfig",
    "StrategyConfig",
    "TrainingConfig",
    "load_config",
    "save_config",
]
