"""End-to-end smoke tests using the synthetic dataset.

These tests cover the full graph (data -> model -> Trainer -> Evaluator) and
the federated client interface (without launching a Flower simulation, which
adds Ray bring-up cost). They run in well under a minute on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running ``pytest tests/`` without ``pip install -e .``.
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _make_cfg(image_size: int = 32, num_samples: int = 64, num_clients: int = 2):
    from fedmammo.configs.schema import (
        AugmentationConfig,
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        LossConfig,
        ModelConfig,
        OptimizerConfig,
        PartitioningConfig,
        SchedulerConfig,
        StrategyConfig,
        TrainingConfig,
    )

    cfg = ExperimentConfig(
        name="smoke",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(
            name="synthetic",
            image_size=image_size,
            grayscale=True,
            num_classes=2,
            batch_size=8,
            num_workers=0,
            synthetic_num_samples=num_samples,
            balance_classes=True,
        ),
        partitioning=PartitioningConfig(
            scheme="dirichlet", alpha=0.5, min_per_client=4, max_retries=50
        ),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        training=TrainingConfig(
            epochs=1,
            local_epochs=1,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            scheduler=SchedulerConfig(name="none"),
            augmentation=AugmentationConfig(),
            loss=LossConfig(name="ce", auto_class_weights=True),
        ),
        federated=FederatedConfig(
            num_clients=num_clients,
            rounds=1,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            strategy=StrategyConfig(name="fedavg"),
        ),
    )
    return cfg


def test_imports() -> None:
    """Importing the package and submodules must not raise."""
    import fedmammo  # noqa: F401
    import fedmammo.configs  # noqa: F401
    import fedmammo.datasets  # noqa: F401
    import fedmammo.evaluation  # noqa: F401
    import fedmammo.federated  # noqa: F401
    import fedmammo.models  # noqa: F401
    import fedmammo.training  # noqa: F401
    import fedmammo.utils  # noqa: F401


def test_synthetic_dataset_shape() -> None:
    from fedmammo.datasets.synthetic import SyntheticMammographyDataset
    from fedmammo.datasets.transforms import build_transforms
    from fedmammo.configs.schema import AugmentationConfig

    train_tx, eval_tx = build_transforms(32, AugmentationConfig(), grayscale=True)
    ds = SyntheticMammographyDataset(
        num_samples=16, image_size=32, grayscale=True, seed=0, transform=train_tx
    )
    assert len(ds) == 16
    x, y = ds[0]
    assert x.shape == (1, 32, 32), f"unexpected shape {x.shape}"
    assert int(y) in (0, 1)
    counts = ds.class_counts()
    assert sum(counts.values()) == 16


def test_partitioning_iid_and_dirichlet() -> None:
    import numpy as np

    from fedmammo.datasets.partitioning import partition_indices

    labels = np.repeat([0, 1], 32)
    iid = partition_indices(labels, num_clients=4, scheme="iid", seed=1)
    assert len(iid) == 4
    assert sum(len(c) for c in iid) == labels.size

    dir_parts = partition_indices(
        labels, num_clients=4, scheme="dirichlet", alpha=0.5, min_per_client=4, seed=1
    )
    assert len(dir_parts) == 4
    assert min(len(c) for c in dir_parts) >= 4


def test_centralized_one_epoch_cpu() -> None:
    """End-to-end: build dataset, model, trainer, run one epoch on CPU."""
    import torch

    from fedmammo.datasets import build_dataloader, build_dataset
    from fedmammo.evaluation import Evaluator
    from fedmammo.models import build_model
    from fedmammo.training import Trainer, build_loss, build_optimizer

    cfg = _make_cfg(image_size=32, num_samples=32, num_clients=2)
    datasets = build_dataset(cfg)
    train_loader = build_dataloader(datasets["train"], batch_size=8, num_workers=0)
    val_loader = build_dataloader(
        datasets["val"], batch_size=8, num_workers=0, shuffle=False
    )

    device = torch.device("cpu")
    model = build_model(cfg.model).to(device)
    criterion = build_loss(cfg.training.loss, train_labels=datasets["train"].labels).to(device)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    trainer = Trainer(model, optimizer, criterion, device, log_tag="smoke")
    evaluator = Evaluator(model, device=device)

    stats = trainer.train_one_epoch(train_loader, epoch=0)
    assert stats["samples"] > 0
    metrics = evaluator.evaluate(val_loader, criterion=criterion)
    for key in ("accuracy", "precision", "recall", "f1", "sensitivity", "specificity"):
        assert key in metrics


def test_federated_client_fit_evaluate() -> None:
    """Exercise FedMammoClient.fit / .evaluate without spinning up Flower."""
    import torch

    from fedmammo.datasets import build_dataset, partition_indices
    from fedmammo.federated.client import FedMammoClient
    from fedmammo.federated.param_utils import state_dict_to_ndarrays
    from fedmammo.models import build_model

    cfg = _make_cfg(image_size=32, num_samples=64, num_clients=2)
    datasets = build_dataset(cfg)
    parts = partition_indices(
        datasets["train"].labels, num_clients=2, scheme="iid", seed=0
    )
    sub_train = datasets["train"].subset(parts[0])
    device = torch.device("cpu")

    client = FedMammoClient(
        client_id=0,
        cfg=cfg,
        train_dataset=sub_train,
        val_dataset=datasets["val"],
        device=device,
    )

    # Pull initial params from a fresh model — server pushes them in real FL.
    template = build_model(cfg.model).to(device)
    init_params = state_dict_to_ndarrays(template)

    new_params, n_samples, fit_metrics = client.fit(init_params, {"local_epochs": 1})
    assert len(new_params) == len(init_params)
    assert n_samples > 0
    assert "train_loss" in fit_metrics

    loss, n_eval, eval_metrics = client.evaluate(new_params, {})
    assert n_eval > 0
    assert "roc_auc" in eval_metrics or "accuracy" in eval_metrics
    assert isinstance(loss, float)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
